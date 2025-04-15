import os
from ..config import Config
from PIL import Image
import json

extensions = set()
image_sizes = {}

def list_extensions(dir):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isdir(path):
            list_extensions(path)
        else:
            ext = file.split('.')[-1]
            if ext != file:
                extensions.add(ext)


def list_image_sizes(dir):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isdir(path):
            list_image_sizes(path)
        else:
            if file.endswith('.png'):
                image = Image.open(path)
                image_sizes[image.size] = image_sizes.get(image.size, 0) + 1
                if image.size == (2471, 3531):
                    print(path)

def number_of_images(dir):
    count = 0
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isdir(path):
            count += number_of_images(path)
        else:
            if file.endswith('.png'):
                count += 1
    return count

def number_of_unique_sentences(sentence_path):
    with open(sentence_path, 'r') as f:
        sentences = json.load(f)
    return len(sentences)

def average_sentence_length(sentence_path):
    with open(sentence_path, 'r') as f:
        sentences = json.load(f)
    total_length = 0
    for sentence in sentences.values():
        total_length += len(sentence.split())
    return total_length / len(sentences)

list_extensions(Config.Paths.data_path)
list_image_sizes(Config.Paths.data_path)
print(f"The data directory contains the following file extensions: {Config.Colors.blue}{extensions}{Config.Colors.reset}")
print(f"The data directory contains the following image sizes, with the number of images of that size: {Config.Colors.blue}{image_sizes}{Config.Colors.reset}")
print(f"There are {Config.Colors.blue}{number_of_images(Config.Paths.data_path)}{Config.Colors.reset} images in the data directory")
print(f"There are {Config.Colors.blue}{number_of_unique_sentences(Config.Paths.sentences_path)}{Config.Colors.reset} unique sentences in the sentences file")
print(f"The average sentence length is {Config.Colors.blue}{average_sentence_length(Config.Paths.sentences_path)}{Config.Colors.reset} words")