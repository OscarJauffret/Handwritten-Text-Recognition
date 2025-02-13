import torch
from CRNN import CRNN
from dataset import HandwritingDataset
from config import Config
from skimage.io import imread
import numpy as np
import os

# Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(Config.Model.alphabet) + 1  # +1 pour le blank CTC
crnn = CRNN(num_classes=num_classes).to(device)

crnn.load_state_dict(torch.load("model100epochs.pth", map_location=device))  # Charger le modèle
crnn.eval()  # Mettre en mode évaluation

def preprocess_image(image_path):
    img = imread(image_path, as_gray=True)
    img = img.astype(np.float32) / 255.0  # Normaliser
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # Ajouter batch et channel dim
    return img.to(device)

# Charger une image de test
image_path = "r06-137-04.png"
image_path = os.path.join(Config.Paths.test_lines, image_path)
image = preprocess_image(image_path)

with torch.no_grad():
    outputs = crnn(image)
    outputs = outputs.log_softmax(2)  # Nécessaire pour CTC
    _, indices = torch.max(outputs, 2)  # Prendre les indices max

# Décoder la sortie
def decode_output(indices):
    decoded_text = ""
    prev_idx = None
    for idx in indices.squeeze(1).cpu().numpy().flatten():  # 🔥 Assure un tableau 1D
        if idx != 0 and (prev_idx is None or idx != prev_idx):  # 🔥 Correction
            decoded_text += Config.Model.alphabet[idx - 1]  # Décodage
        prev_idx = idx
    return decoded_text


# Afficher le résultat
predicted_text = decode_output(indices)
print(f"Texte prédit : {predicted_text}")
