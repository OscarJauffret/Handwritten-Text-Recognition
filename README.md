# 📝 Handwritten Text Recognition

This project focuses on recognizing handwritten text using deep learning, trained on the IAM dataset.

> [!NOTE]
> The project is still in progress.

## 🛠️ Requirements

- Python

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📋 Workflow Overview

### 1. 📁 Data Preprocessing

- Load the IAM dataset
- Extract meta-information (text data) from image annotations:
  ```bash
  python code/dump_XML_to_dir.py
  ```

### 2. 🔀 Dataset Splitting

- Split the dataset into training and validation sets:
  ```bash
  python code/split.py
  ```

### 3. ✂️ Word-Level Image Splitting

- Split each image into individual words.
- Works on all images in `train/`, `validate/`, and `test/` folders.
- Saves the words into the `words/` directory:
  ```bash
  python code/split_images.py
  ```
  
### 4. 🖊️ Label Generation
- Generate labels for the words, using the xml files.
  ```bash
  python code/generate_labels.py
  ```

---
