import torch

from src.model.CRNN import CRNN
from .utils.utils import select_device
from src.model.dataset import HandwritingDataset
from torch.utils.data import DataLoader
from src.model.trainer import Trainer

from .config import Config

def train():
    device = select_device()

    dataset = HandwritingDataset(Config.Paths.train_words, Config.Paths.train_labels, device)
    # Training a models on a single image at a time would be too slow, so we create a batch of 8 images which will be fed to the models at once.
    # A batch is a mini-group that will do a single update of the models's weights.
    # A batch size of 8 is a good compromise between speed and not too much memory usage.
    train_loader = DataLoader(dataset, batch_size=Config.Model.batch_size, shuffle=True, num_workers=Config.Model.num_workers)  # Shuffling avoids the models learning the order of the images

    val_dataset = HandwritingDataset(Config.Paths.validate_words, Config.Paths.validate_labels, device)
    val_loader = DataLoader(val_dataset, batch_size=Config.Model.batch_size, shuffle=False, num_workers=Config.Model.num_workers)


    # Load the models, loss function, and optimizer
    num_classes = len(Config.Model.alphabet) + 1  # +1 for the blank character
    model = CRNN(num_classes=num_classes).to(device)

    if Config.Paths.resume_checkpoint is not None:
        print(f"Loading model checkpoint from {Config.Paths.resume_checkpoint}...")
        model.load_state_dict(torch.load(Config.Paths.resume_checkpoint, map_location=device))

    trainer = Trainer(model, lr=0.0001, patience=5, device=device)
    
    # Training
    trainer.train(train_loader, val_loader, epochs=Config.Model.epochs)

if __name__ == '__main__':
    train()