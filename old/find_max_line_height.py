from config import Config
import os
from old.separate_into_lines import find_lines
from skimage.io import imread
import xml.etree.ElementTree as gfg
from tqdm import tqdm

max = 0

files = os.listdir(Config.Paths.train)
for file in tqdm(files, desc="Processing images", unit="file"):
    lines = find_lines(imread(os.path.join(Config.Paths.train, file)))
    for start, end in lines:
        if end - start > max:
            max = end - start
            print(max, file)


root = gfg.Element("information")
size = gfg.Element("size")
root.append(size)
max_element = gfg.SubElement(size, "max-line-height")
max_element.text = str(max)

tree = gfg.ElementTree(root)

with open(Config.Paths.sentences_sizes_path, 'wb') as f:
    tree.write(f)


