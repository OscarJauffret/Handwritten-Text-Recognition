import os
import xml.etree.ElementTree as ET
from config import Config

def generate_labels(image_folder, label_folder, mode='lines', xml_folder=Config.Paths.images_meta_info_path):
    """
    Generate labels for the images in the image_folder and save them in the label_folder
    :param image_folder: The folder containing the images
    :param label_folder: The folder to save the labels
    :param mode: The mode to use to extract the labels. Can be 'lines' or 'words'
    :param xml_folder: The folder containing the XML files
    :return: None
    """
    os.makedirs(label_folder, exist_ok=True)

    for filename in os.listdir(image_folder):   # Iterate over the files in the image folder
        if not filename.endswith(".png"):
            continue
        base_name = filename.split(".")[0]
        xml_path = os.path.join(xml_folder, base_name + ".xml") # Get the path to the corresponding XML file
        if not os.path.exists(xml_path):
            print(f"{Config.Colors.warning}Warning: Could not find XML file for {base_name}{Config.Colors.reset}")
            continue
        label_dict = extract_from_xml(xml_path, mode)    # Extract the labels from the XML file
        for id, text in label_dict.items():
            with open(os.path.join(label_folder, f"{id}.txt"), "w") as f:      # Save the text in a file with the id as the name
                f.write(text)

def extract_from_xml(xml_file, mode='lines'):
    if mode == 'lines':
        return extract_lines_from_xml(xml_file)
    elif mode == 'words':
        return extract_words_from_xml(xml_file)
    else:
        raise ValueError("Mode invalide. Utilisez 'lines' ou 'words'.")


def extract_lines_from_xml(xml_file):
    """
    Extract the lines from an XML file
    :param xml_file: The path to the XML file
    :return: A dictionary with the line_id as the key and the text as the value
    """
    with open(xml_file, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        lines = {}
        handwritten_part = root.find('handwritten-part')    # Find the handwritten part of the XML file

        if handwritten_part is not None:
            for line in handwritten_part.findall(".//line"):    # For each line in the handwritten part
                text = ''
                line_id = line.get('id')                        # Get the line_id
                for word in line.findall(".//word"):            # For each word in the line
                    text += word.get('text') + ' '              # Add the text of the word to the line text
                lines[line_id] = text.strip()
    return lines

def extract_words_from_xml(xml_file):
    """
    Extract the words from an XML file
    :param xml_file: The path to the XML file
    :return: A dictionary with the word_id as the key and the text as the value
    """
    with open(xml_file, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        words = {}
        handwritten_part = root.find('handwritten-part')    # Find the handwritten part of the XML file

        if handwritten_part is not None:
            for line in handwritten_part.findall(".//line"):    # For each line in the handwritten part
                for word in line.findall(".//word"):            # For each word in the line
                    words[word.get('id')] = word.get('text')
    return words

if __name__ == "__main__":
    mode = 'words'
    generate_labels(Config.Paths.train_images, Config.Paths.train_labels, mode)
    generate_labels(Config.Paths.test_images, Config.Paths.test_labels, mode)
    generate_labels(Config.Paths.validate_images, Config.Paths.validate_labels, mode)