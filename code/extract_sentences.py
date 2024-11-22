import os
import json
import pytesseract
import re
from config import Config
from tqdm import tqdm
from separate_page_sections import separate_header
from skimage.io import imread

data = {}

def extract_text_from_image(image_path):
    file_match = re.search(Config.file_name_regex, image_path, re.IGNORECASE)
    if not file_match:
        print(f"{Config.Colors.error}No file_name match found for {image_path}{Config.Colors.reset}")
        return False

    file_header = file_match.group(1)
    if file_header in data:
        return True
    else:
        image = imread(image_path)
        lines_positions = separate_header(image)
        text_to_extract = image[lines_positions[0]:lines_positions[1]]
        text = pytesseract.image_to_string(text_to_extract)
        text = text.replace("\n", " ")
        # Remove tailing spaces
        data[file_header] = re.sub(r"\s+$", "", text)
        return True


def extract_text_from_images(data_path):
    files = []
    failed_files = []
    for root, _, file_names in os.walk(data_path):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))

    for file_path in tqdm(files, desc="Processing images", unit="file"):
        if not extract_text_from_image(file_path):
            failed_files.append(file_path)

    if failed_files:
        print(f"{Config.Colors.error}Failed to extract text from the following files:{Config.Colors.reset}")
        for file in failed_files:
            print(file)
        print(f"{Config.Colors.blue}Success rate: {1 - len(failed_files) / len(files)}{Config.Colors.reset}")


extract_text_from_images(Config.Paths.data_path)

with open(Config.Paths.sentences_path, "w") as f:
    f.write(json.dumps(data))

for key, value in data.items():
    with open(os.path.join(Config.Paths.individual_sentences_path, f"{key}.txt"), "w") as f:
        f.write(value)