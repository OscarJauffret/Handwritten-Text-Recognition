"""
The purpose of this script is to split an image into multiple images based on the lines in the image.
Each line in the image will be saved as a separate image.
"""

from config import Config
from separate_into_lines import find_lines
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import os

def extract_lines(image_path, save_path=Config.Paths.train_lines):
    """
    Extracts the lines from an image and saves them as separate images
    :param image_path: the path to the image
    :param save_path: the path to save the lines
    :return: a list of the extracted lines
    """
    img = imread(image_path, as_gray=True)
    filename = os.path.basename(image_path).split('.')[0]
    os.makedirs(save_path, exist_ok=True)

    line_positions = find_lines(img)
    line_images = []
    for i, (start, end) in enumerate(line_positions):
        line_img = img[start:end]
        line_img = downsample_image(line_img, Config.Data.line_height, Config.Data.max_line_width)

        line_filename = f"{filename}-{i:02}.png"        # Change the name of the file to include the line number with two digits
        imsave(os.path.join(save_path, line_filename), line_img)    # Save the line image
        line_images.append(line_img)
    return line_images

def downsample_image(image, target_height=Config.Data.line_height, max_width=Config.Data.max_line_width):
    """
    Downsamples an image to a target height while keeping the aspect ratio and ensuring the width does not exceed a maximum value
    :param image: the image to downsample
    :param target_height: the target height of the image
    :param max_width: the maximum width of the image
    :return: the downsampled image
    """
    original_height, original_width = image.shape
    scale = target_height / original_height
    new_width = int(original_width * scale)

    if new_width > max_width:
        scale = max_width / original_width
        new_width = max_width
        target_height = int(original_height * scale)

    image_resized = resize(image, (target_height, new_width), anti_aliasing=True)
    return (image_resized * 255).astype(np.uint8)


if __name__ == "__main__":
    files = os.listdir(Config.Paths.train_images)
    for file in files:
        extract_lines(os.path.join(Config.Paths.train_images, file), Config.Paths.train_lines)
    files = os.listdir(Config.Paths.test_images)
    for file in files:
        extract_lines(os.path.join(Config.Paths.test_images, file), Config.Paths.test_lines)
    files = os.listdir(Config.Paths.validate_images)
    for file in files:
        extract_lines(os.path.join(Config.Paths.validate_images, file), Config.Paths.validate_lines)