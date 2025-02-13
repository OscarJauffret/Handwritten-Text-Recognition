from config import Config
import xml.etree.ElementTree as ET
import os

def get_line_bounds(image_path):
    """
    Get the line bounds for the image at the given path
    :param image_path: the path to the image
    :return: a list of tuples with the start and end of each line
    """
    base_name = os.path.basename(image_path)
    xml_path = os.path.join(Config.Paths.images_meta_info_path, base_name.replace('.png', '.xml'))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    handwritten_part = root.find('handwritten-part')  # Find the handwritten part of the XML file
    for line in handwritten_part.findall(".//line"):    # For each line in the handwritten part
        start = int(line.get('asy'))
        end = int(line.get('dsy'))
        lines.append((start, end))

    return lines

if __name__ == '__main__':
    image_path = os.path.join(Config.Paths.train_images, 'a01-000u.png')
    lines = get_line_bounds(image_path)
    print(lines)