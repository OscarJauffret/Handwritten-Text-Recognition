import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import textwrap
import editdistance
import argparse

from skimage.io import imread, imsave

from .model.CRNN import CRNN
from .config import Config
from .utils.utils import decode_output, select_device, grayscale
from .init.split_images import downsample_image

arg_parser = argparse.ArgumentParser()
group = arg_parser.add_mutually_exclusive_group()
group.add_argument('--fullpage', action='store_true', help='Whether to test on a full page or a random sample')
group.add_argument('--custom', type=str, help='Path to custom image you want to test')
group.add_argument('--test', action='store_true', help='Evaluate on the full test set and compute CER.')
group.add_argument('--wer', action='store_true', help='Compute WER on the full-page forms in the test set and validation set')
arg_parser.add_argument('--save', action='store_true', help='Whether to save the image')

class Inferer:
    def __init__(self, model_path, test_images_folder, test_words_folder, labels_folder, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = len(Config.Model.alphabet) + 1  # +1 for CTC blank
        self.model = CRNN(num_classes, Config.Model.hidden_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.test_images_folder = test_images_folder
        self.test_words_folder = test_words_folder
        self.labels_folder = labels_folder

    def preprocess_image(self, image_path):
        img = imread(image_path, as_gray=True)
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        return img.to(self.device)

    def calculate_cer(self, prediction, ground_truth):
        return editdistance.eval(prediction, ground_truth) / max(len(ground_truth), 1)

    def calculate_wer(self, prediction, ground_truth):
        pred_words = prediction.strip().split()
        gt_words = ground_truth.strip().split()
        wer = editdistance.eval(pred_words, gt_words) / max(len(gt_words), 1)
        return wer

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(image)
            outputs = outputs.log_softmax(2)
        prediction = decode_output(outputs)[0]
        return prediction

    def predict_full_page(self, base_name, words_folder, labels_folder):
        # Find all corresponding word images
        word_images = sorted([
            f for f in os.listdir(words_folder) if f.startswith(base_name) and f.endswith('.png')
        ])
        reconstructed_text = []
        ground_truth_text = []
        for word_file in word_images:
            word_path = os.path.join(words_folder, word_file)
            prediction = self.predict(word_path)
            reconstructed_text.append(prediction)
            ground_truth = self.get_ground_truth(word_path, labels_folder)
            ground_truth_text.append(ground_truth)
        reconstructed_sentence = ' '.join(reconstructed_text)
        ground_truth_sentence = ' '.join(ground_truth_text)

        return ground_truth_sentence, reconstructed_sentence

    def get_ground_truth(self, image_path, labels_folder):
        label_file_name = image_path.split("/")[-1].split(".")[0] + ".txt"
        label_path = os.path.join(labels_folder, label_file_name)
        with open(label_path, 'r', encoding='utf-8') as f:
            ground_truth = f.readline().strip()
        return ground_truth

    def plot_prediction(self, image_path, prediction, ground_truth, cer, save_plot, output_folder, file, fullpage=False):
        img = imread(image_path, as_gray=True)

        if fullpage:
            figsize = (12, 8)
        else:
            height, width = img.shape
            dpi = 100
            figsize = (width / dpi + 2, height / dpi + 2)

        plt.figure(figsize=figsize)
        plt.imshow(img, cmap='gray', aspect='equal')
        plt.axis('off')
        plt.title(f"CER: {cer * 100:.2f}%")
        plt.subplots_adjust(left=0.15, bottom=0.05, right=0.85, top=0.95, wspace=0.05, hspace=0.05)

        # Wrap long texts
        wrapped_prediction = "\n".join(textwrap.wrap(prediction, width=150))
        wrapped_ground_truth = "\n".join(textwrap.wrap(ground_truth, width=150))
        plt.suptitle(f"Prediction:\n{wrapped_prediction}\n\nGround Truth:\n{wrapped_ground_truth}", fontsize=10)

        plt.tight_layout()

        if save_plot:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.savefig(os.path.join(output_folder, f"{file.split('.')[0]}_qualitative_test.png"))
            plt.close()
        else:
            plt.show()

    def test_random_samples(self, num_samples=5, save_plots=False, output_folder="plots", random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)

        image_files = [f for f in os.listdir(self.test_words_folder) if f.endswith('.png')]
        selected_files = random.sample(image_files, min(num_samples, len(image_files)))

        if save_plots and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in selected_files:
            img_path = os.path.join(self.test_words_folder, file)
            prediction = self.predict(img_path)
            ground_truth = self.get_ground_truth(img_path, self.labels_folder)

            cer = self.calculate_cer(prediction, ground_truth)

            self.plot_prediction(img_path, prediction, ground_truth, cer, save_plots, output_folder, file)

    def test_full_page(self, save_plot=False, output_folder="plots", random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)

        # Pick a random full page image
        full_page_files = [f for f in os.listdir(self.test_images_folder) if f.endswith('.png')]
        selected_file = random.choice(full_page_files)
        base_name = selected_file.split(".")[0]

        ground_truth_sentence, reconstructed_sentence = self.predict_full_page(base_name, self.test_words_folder, self.labels_folder)
        cer = self.calculate_cer(reconstructed_sentence, ground_truth_sentence)
        wer = self.calculate_wer(reconstructed_sentence, ground_truth_sentence)
        print(f"WER on {selected_file}: {wer * 100:.2f}%")

        # Plot the full page
        self.plot_prediction(os.path.join(self.test_images_folder, selected_file), reconstructed_sentence,
                              ground_truth_sentence, cer, save_plot, output_folder, selected_file, True)

    def test_custom(self, image_path, temp_path, save_plot=False, output_folder="plots"):
        img = imread(image_path)
        img = grayscale(img)
        downsampled_img = downsample_image(img, Config.Data.word_height, Config.Data.word_width)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        downsampled_image_path = os.path.join(temp_path, f"{os.path.basename(image_path)}_downsampled.png")
        imsave(downsampled_image_path, downsampled_img) # Save the downsampled image

        prediction = self.predict(downsampled_image_path)

        height, width = img.shape
        dpi = 100
        figsize = (width / dpi + 2, height / dpi + 2)

        plt.figure(figsize=figsize)
        plt.imshow(img, cmap='gray', aspect='equal')
        plt.title(f"Prediction: {prediction}")
        plt.axis('off')

        plt.tight_layout()

        if save_plot:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.savefig(os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_qualitative_test.png"))
            plt.close()
        else:
            plt.show()

        os.remove(downsampled_image_path)

    def test_set(self):
        image_files = [f for f in os.listdir(self.test_words_folder) if f.endswith('.png')]
        total_cer = 0
        count = 0

        for file in image_files:
            img_path = os.path.join(self.test_words_folder, file)
            prediction = self.predict(img_path)
            ground_truth = self.get_ground_truth(img_path, self.labels_folder)
            cer = self.calculate_cer(prediction, ground_truth)
            total_cer += cer
            count += 1

        avg_cer = total_cer / max(count, 1)
        print(f"Average CER on test set: {avg_cer * 100:.2f}%")

    def full_page_wer(self, folder, save_plot=False, output_folder="plots"):
        forms_folder = os.path.join(folder, "images")
        words_folder = os.path.join(folder, "words")
        labels_folder = os.path.join(folder, "labels")

        full_page_files = [f for f in os.listdir(forms_folder) if f.endswith('.png')]
        total_wer = 0
        total_words = 0

        for selected_file in full_page_files:
            base_name = selected_file.split(".")[0]
            reconstructed_sentence, ground_truth_sentence = self.predict_full_page(base_name, words_folder, labels_folder)

            wer = self.calculate_wer(reconstructed_sentence, ground_truth_sentence)
            total_wer += wer
            total_words += 1

            if save_plot:
                self.plot_prediction(os.path.join(forms_folder, selected_file), reconstructed_sentence,
                                     ground_truth_sentence, wer, save_plot, output_folder, selected_file, True)

        avg_wer = total_wer / max(total_words, 1)
        print(f"Average WER on full-page forms in {forms_folder}: {avg_wer * 100:.2f}%")


if __name__ == '__main__':
    args = arg_parser.parse_args()
    model_path = os.path.join(Config.Paths.models_path, "temp_best.pth")
    test_images_folder = Config.Paths.test_images
    test_words_folder = Config.Paths.test_words
    labels_folder = Config.Paths.test_labels
    device = select_device()
    inferer = Inferer(model_path, test_images_folder, test_words_folder, labels_folder, device)
    if args.test:
        inferer.test_set()
        exit()
    if args.wer:
        inferer.full_page_wer(Config.Paths.test, save_plot=args.save)
        inferer.full_page_wer(Config.Paths.validate, save_plot=args.save)
        exit()
    if not args.custom:
        if args.fullpage:
            inferer.test_full_page(save_plot=args.save)
        else:
            inferer.test_random_samples(num_samples=5, save_plots=args.save)
    else:
        inferer.test_custom(args.custom, "tmp", save_plot=args.save)

