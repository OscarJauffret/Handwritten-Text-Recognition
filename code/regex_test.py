from config import Config
import re
from PIL import Image
import pytesseract

def extract_text_from_image(image_path):

    file_match = re.search(Config.file_name_regex, image_path, re.IGNORECASE)
    if not file_match:
        print(f"{Config.Colors.error}No file_name match found for {image_path}{Config.Colors.reset}")
        return False

    file_header = file_match.group(1)
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    body = text.split("\n\n")[1]
    print(body)

extract_text_from_image("../data/017/b01-057.png")