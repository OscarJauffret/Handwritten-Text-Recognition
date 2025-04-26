import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from skimage.io import imread
import editdistance

from .model.CRNN import CRNN
from .config import Config
from .utils.utils import decode_output

class Inferer:
    def __init__(self, model_path, test_folder, labels_folder, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = len(Config.Model.alphabet) + 1  # +1 for CTC blank
        self.model = CRNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.test_folder = test_folder
        self.labels_folder = labels_folder

    def preprocess_image(self, image_path):
        img = imread(image_path, as_gray=True)
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        return img.to(self.device)

    def calculate_cer(self, prediction, ground_truth):
        return editdistance.eval(prediction, ground_truth) / max(len(ground_truth), 1)

    def test_random_samples(self, num_samples=5, save_plots=False, output_folder="plots", random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)

        image_files = [f for f in os.listdir(self.test_folder) if f.endswith('.png')]
        selected_files = random.sample(image_files, min(num_samples, len(image_files)))

        if save_plots and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in selected_files:
            img_path = os.path.join(self.test_folder, file)
            image = self.preprocess_image(img_path)

            with torch.no_grad():
                outputs = self.model(image)
                outputs = outputs.log_softmax(2)

            prediction = decode_output(outputs)[0]
            label_file_name = file.split(".")[0] + ".txt"
            label_path = os.path.join(self.labels_folder, label_file_name)
            with open(label_path, 'r', encoding='utf-8') as f:
                ground_truth = f.readline().strip()

            cer = self.calculate_cer(prediction, ground_truth)

            img = imread(img_path, as_gray=True)

            plt.figure()
            plt.imshow(img, cmap='gray', aspect='auto')
            plt.axis('off')
            plt.title(f"Prediction: {prediction}\nGround Truth: {ground_truth}\nCER: {cer:.4f}")
            plt.tight_layout()

            if save_plots:
                plt.savefig(os.path.join(output_folder, f"{file.split('.')[0]}_qualitative_test.png"))
                plt.close()
            else:
                plt.show()


if __name__ == '__main__':
    model_path = os.path.join(Config.Paths.models_path, "first_model_1.16CTC.pth")
    test_folder = Config.Paths.test_words
    labels_folder = Config.Paths.test_labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferer = Inferer(model_path, test_folder, labels_folder, device)
    inferer.test_random_samples(num_samples=5)