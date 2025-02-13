"""
The purpose of this script is to split an image into multiple images based on the lines in the image.
Each line in the image will be saved as a separate image.
"""

from config import Config
from code.separate_into_lines import get_line_bounds
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

    line_positions = get_line_bounds(image_path)
    line_images = []
    for i, (start, end) in enumerate(line_positions):
        line_img = img[start:end]
        line_img = downsample_image(line_img, Config.Data.line_height, Config.Data.line_width)

        line_filename = f"{filename}-{i:02}.png"        # Change the name of the file to include the line number with two digits
        imsave(os.path.join(save_path, line_filename), line_img)    # Save the line image
        line_images.append(line_img)
    return line_images

def downsample_image(image, target_height=Config.Data.line_height, target_width=Config.Data.line_width):
    """
    Resize an image while keeping the aspect ratio and adding padding.

    :param image: The input image as a NumPy array.
    :param target_height: The target height.
    :param target_width: The target width.
    :return: The resized image with padding.
    """
    original_height, original_width = image.shape
    scale = target_height / original_height
    new_width = int(original_width * scale)

    if new_width > target_width:  # If the width is too large, then resize based on the width
        scale = target_width / original_width
        new_width = target_width
        new_height = int(original_height * scale)
    else:
        new_height = target_height

    image_resized = resize(image, (new_height, new_width), anti_aliasing=True)
    image_resized = (image_resized * 255).astype(np.uint8)

    padded_image = np.ones((target_height, target_width), dtype=np.uint8) * 255  # Create a white image

    # Add the resized image to the padded image in the top left corner
    padded_image[:new_height, :new_width] = image_resized

    return padded_image

if __name__ == "__main__":
    for dataset_type, img_path, save_path in [
        ("train", Config.Paths.train_images, Config.Paths.train_lines),
        ("test", Config.Paths.test_images, Config.Paths.test_lines),
        ("validate", Config.Paths.validate_images, Config.Paths.validate_lines),
    ]:
        print(f"Processing {dataset_type} images...")
        files = os.listdir(img_path)
        for file in files:
            extract_lines(os.path.join(img_path, file), save_path)
        print(f"{dataset_type} set processed.")