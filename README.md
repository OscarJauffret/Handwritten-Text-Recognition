# 📝 Handwritten Text Recognition

This project focuses on recognizing handwritten text using deep learning, trained on the IAM dataset.

## 🛠️ Requirements

- Python

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📋 Steps

Run the following script to automatically perform all preprocessing steps:

```bash
python -m src.init.init_project
```

This script includes:
- XML metadata extraction
- Train/validation/test split
- Word-level image cropping
- Label generation

Then, you can train the model
```bash
python -m src.train
```

Finally, you can test the model with
```bash
python -m src.test [--fullpage | --custom CUSTOM] [--save]
```
