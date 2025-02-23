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


def get_words_bounds(image_path):
    base_name = os.path.basename(image_path)
    xml_path = os.path.join(Config.Paths.images_meta_info_path, base_name.replace('.png', '.xml'))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    words = []

    handwritten_part = root.find('handwritten-part')
    for line in handwritten_part.findall(".//line"):  # For each line in the handwritten part
        for word in line.findall(".//word"):
            id = word.get('id')
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0

            for cmp in word.findall(".//cmp"):
                x = int(cmp.get('x'))
                y = int(cmp.get('y'))
                w = int(cmp.get('width'))
                h = int(cmp.get('height'))

                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

            if x_min == float('inf') or y_min == float('inf') or x_max == 0 or y_max == 0:
                continue
            words += [(id, (x_min, y_min), (x_max, y_max))]

    return words

if __name__ == '__main__':
    image_path = os.path.join(Config.Paths.train_images, 'a01-000u.png')
    words = get_words_bounds(image_path)
    print(words)
