import torch
import os

from src.model.CRNN import CRNN
from .utils.utils import select_device
from src.model.dataset import HandwritingDataset
from torch.utils.data import DataLoader
from src.model.trainer import Trainer
from argparse import ArgumentParser

from .config import Config

arg_parser = ArgumentParser()
arg_parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file to resume training from')
args = arg_parser.parse_args()

def train():
    device = select_device()

    dataset = HandwritingDataset(Config.Paths.train_words, Config.Paths.train_labels, device, augment=Config.Model.augmentation)
    # Training a models on a single image at a time would be too slow, so we create a batch of 8 images which will be fed to the models at once.
    # A batch is a mini-group that will do a single update of the models' weights.
    # A batch size of 8 is a good compromise between speed and not too much memory usage.
    train_loader = DataLoader(dataset, batch_size=Config.Model.batch_size, shuffle=True, num_workers=Config.Model.num_workers)  # Shuffling avoids the models learning the order of the images

    val_dataset = HandwritingDataset(Config.Paths.validate_words, Config.Paths.validate_labels, device, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.Model.batch_size, shuffle=False, num_workers=Config.Model.num_workers)


    # Load the models, loss function, and optimizer
    num_classes = len(Config.Model.alphabet) + 1  # +1 for the blank character
    model = CRNN(num_classes, Config.Model.hidden_size).to(device)

    if args.checkpoint is not None:
        checkpoint_path = os.path.join(Config.Paths.models_path, args.checkpoint)
        print(f"Loading model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

        # Load only compatible layers
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"✅ Loaded {len(pretrained_dict)} layers from checkpoint.")
        print(f"❌ Skipped {len(checkpoint) - len(pretrained_dict)} incompatible layers.")

        # Freeze first two convolutional blocks
        for layer in list(model.cnn.children())[:6]:  # Conv2d, BatchNorm2d, ReLU, MaxPool2d (first block) + next Conv2d, BatchNorm2d
            for param in layer.parameters():
                param.requires_grad = False

    else:
        print("No checkpoint provided, starting from scratch...")

    trainer = Trainer(model, lr=Config.Model.learning_rate, patience=Config.Model.patience, device=device)
    
    # Training
    trainer.train(train_loader, val_loader, epochs=Config.Model.epochs)

if __name__ == '__main__':
    train()