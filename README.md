# Handwritten Text Recognition

## Introduction
Recognition of handwritten text using deep learning. The model is trained on the IAM dataset.

> [!NOTE]
> The project is still in progress.

## Requirements
- Python
```shell
pip install -r requirements.txt
```
## Steps (up to now)
### Data Preprocessing
   1. Load the IAM dataset
   2. Load the meta-information about the images to get the text
   ```shell
   python code/dump_XML_to_dir.py
   ```
   
### Split the dataset into training and validation sets
   ```shell
   python code/split.py
   ```
### Split each image of the dataset into words
The script will loop over all images in train/validate/test folders and split them into words.
The words will be saved in the words/ folder.
```shell
python code/split_images.py
```

