import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage.io import imread
from config import Config
from matplotlib import pyplot as plt

class HandwritingDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform

        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, img_name)
        label_path = os.path.join(self.label_folder, img_name.replace('.png', '.txt'))

        image = imread(image_path, as_gray=True)
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)

        with open(label_path, 'r', encoding='utf-8') as f:
            text = f.readline().strip()

        return image, text


if __name__ == '__main__':
    image_folder = Config.Paths.train_words
    label_folder = Config.Paths.train_labels
    dataset = HandwritingDataset(image_folder, label_folder)

    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for images, texts in train_loader:
        plt.imshow(images[0].squeeze(0), cmap="gray")
        plt.title(f"Text : {texts[0]}")
        plt.show()
        break