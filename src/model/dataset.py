import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage.io import imread
from ..config import Config
from ..utils.utils import select_device
from matplotlib import pyplot as plt
from tqdm import tqdm

class HandwritingDataset(Dataset):
    def __init__(self, image_folder, label_folder, device, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.device = device
        self.transform = transform

        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

        images = []
        labels = []
        for fname in tqdm(self.image_files, desc="Loading dataset"):
            image_path = os.path.join(self.image_folder, fname)
            label_path = os.path.join(self.label_folder, fname.replace('.png', '.txt'))

            image = imread(image_path, as_gray=True).astype(np.float32) / 255.0
            image = torch.tensor(image).unsqueeze(0)  # [1, H, W]
            images.append(image)

            with open(label_path, 'r', encoding='utf-8') as f:
                text = f.readline().strip()
            labels.append(text)

        self.data = torch.stack(images).to(self.device)  # Shape: [N, 1, H, W]
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == '__main__':
    image_folder = Config.Paths.train_words
    label_folder = Config.Paths.train_labels
    device = select_device()
    dataset = HandwritingDataset(image_folder, label_folder, device)

    train_loader = DataLoader(dataset, batch_size=Config.Model.batch_size, shuffle=True)

    for images, texts in train_loader:
        plt.imshow(images[0].squeeze(0), cmap="gray")
        plt.title(f"Text : {texts[0]}")
        plt.show()
        break