import torch

from src.model.CRNN import CRNN
from .utils.utils import select_device
from src.model.dataset import HandwritingDataset
from torch.utils.data import DataLoader
from src.model.trainer import Trainer

from .config import Config

def train():
    torch.backends.cudnn.benchmark = True

    dataset = HandwritingDataset(Config.Paths.train_words, Config.Paths.train_labels)
    # Training a model on a single image at a time would be too slow, so we create a batch of 8 images which will be fed to the model at once.
    # A batch is a mini-group that will do a single update of the model's weights.
    # A batch size of 8 is a good compromise between speed and not too much memory usage.
    train_loader = DataLoader(dataset, batch_size=Config.Model.batch_size, shuffle=True, num_workers=8)  # Shuffling avoids the model learning the order of the images

    val_dataset = HandwritingDataset(Config.Paths.validate_words, Config.Paths.validate_labels)
    val_loader = DataLoader(val_dataset, batch_size=Config.Model.batch_size, shuffle=False, num_workers=4)

    device = select_device()

    # Load the model, loss function, and optimizer
    num_classes = len(Config.Model.alphabet) + 1  # +1 for the blank character
    model = CRNN(num_classes=num_classes).to(device)
    trainer = Trainer(model, lr=0.0001, patience=5, device=device)
    
    # Training
    torch.autograd.set_detect_anomaly(True) # Will halt training and point to the first place where something goes wrong

    trainer.train(train_loader, val_loader, epochs=Config.Model.epochs)

if __name__ == '__main__':
    train()