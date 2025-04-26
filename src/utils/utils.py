import torch
from src.config import Config


def encode_texts(texts):
    """
    Convert a list of strings to a tensor containing the encoded indices.

    :param texts: List of strings
    :return: Tensor containing the encoded indices
    """
    # This is the alphabet that the models will learn to recognize
    # All the characters that can be recognized. This is the vocabulary of the models, it only recognizes these characters
    # Associate each character with an index. We add 1 because the CTC loss expects the blank character to be at index 0
    char_to_idx = {char: idx + 1 for idx, char in enumerate(Config.Model.alphabet)}
    char_to_idx["<BLANK>"] = 0  # Blank character for CTC loss

    encoded = []
    for text in texts:
        text_encoded = []
        for char in text:
            if char in char_to_idx:
                text_encoded.append(char_to_idx[char])
            else:
                print(f"{Config.Colors.warning}Character '{char}' not in the alphabet{Config.Colors.reset}")
        encoded.extend(text_encoded)  # Add the indices to the list
    return torch.tensor(encoded, dtype=torch.long)


def decode_output(output):
    """
    Decode the output of the models to text.

    :param output: Output of the models
    :return: Decoded text
    """
    char_to_idx = {char: idx + 1 for idx, char in enumerate(Config.Model.alphabet)}
    char_to_idx["<BLANK>"] = 0  # Blank character for CTC loss

    _, indices = torch.max(output, 2) # Get the index with the highest probability
    indices = indices.squeeze(1).cpu().numpy()  # Remove the batch dimension and convert to NumPy

    decoded_text = ""
    prev_idx = None

    for idx in indices:
        if idx != 0 and idx != prev_idx:  # Ignore the blank character and consecutive duplicates
            decoded_text += list(char_to_idx.keys())[list(char_to_idx.values()).index(idx)]  # Convert the index to the corresponding character
        prev_idx = idx

    return decoded_text

def select_device():
    """
    Select and return the appropriate computing device based on hardware availability.

    This function prioritizes the GPU (CUDA) if available, then falls back on
    Metal Performance Shaders (MPS) for Apple devices. If neither is available,
    it defaults to the CPU.

    :return: The selected device (CUDA, MPS, or CPU) according to the hardware availability.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device
