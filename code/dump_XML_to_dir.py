from config import Config
import os
import xml.etree.ElementTree as ET

def extract_text_from_xml(xml_file):
    with open(xml_file, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        text = ''
        machine_text = root.find('machine-printed-part')

        if machine_text is not None:
            for line in machine_text.findall(".//machine-print-line"):
                line_text = line.get('text')
                if line_text:
                    text += line_text + '\n'

        return text


for file in os.listdir(Config.Paths.images_meta_info_path):
    if file.endswith('.xml'):
        xml_file = os.path.join(Config.Paths.images_meta_info_path, file)
        text = extract_text_from_xml(xml_file)
        if not text:
            print(f"{Config.Colors.error}SEVERE: No text extracted from {xml_file}{Config.Colors.reset}")
        with open(os.path.join(Config.Paths.individual_sentences_path, f"{file[:-4]}.txt"), "w") as f:
            f.write(text)
    else:
        print(f"Skipping {file} because it is not an XML file")
