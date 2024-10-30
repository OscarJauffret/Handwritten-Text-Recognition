import os
import json
from PIL import Image
import pytesseract
import re
from config import Config
from tqdm import tqdm

data = {}

def extract_text_from_image(image_path):

    file_match = re.search(Config.file_name_regex, image_path, re.IGNORECASE)
    if not file_match:
        print(f"{Config.Colors.error}No file_name match found for {image_path}{Config.Colors.reset}")
        return False

    file_header = file_match.group(1)
    if file_header in data:
        print(f"{Config.Colors.gray}Already extracted text from {file_header}{Config.Colors.reset}")
        return True
    else:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        body = text.split("\n\n")[1]
        data[file_header] = body.replace("\n", " ")
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
