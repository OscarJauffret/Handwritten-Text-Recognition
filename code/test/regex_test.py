from config import Config
import re
from PIL import Image
import pytesseract
from skimage.filters import threshold_otsu

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    #file_match = re.search(Config.file_name_regex, image_path, re.IGNORECASE)
    #if not file_match:
    #    print(f"{Config.Colors.error}No file_name match found for {image_path}{Config.Colors.reset}")
    #    return False
#
    #file_header = file_match.group(1)
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    body = text.split("\n\n")[1]
    print(text)
    # if len(body.split(" ")) < 30: # If the text is too short, it's probably gibberish, (at least 30 words)
    #     print(f"{Config.Colors.warning}Text too short for {file_header}. Skipping.{Config.Colors.reset}")
    #     body = text.split("\n\n")[2]
    # print(body)

extract_text_from_image("../data/000/a01-077u.png")
extract_text_from_image("../../output/a01-077u_median.png")
extract_text_from_image("../../output/a01-077u_binary.png")