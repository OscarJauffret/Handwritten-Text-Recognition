# Handwritten Text Recognition

## Introduction
Recognition of handwritten text using deep learning. The model is trained on the IAM dataset.

> [!NOTE]
> The project is still in progress.

> [!TODO]
> - Change the magic number 8 in code/separate_into_lines.py

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

### Split the forms into lines
```shell
python code/separate_into_lines.py
```

