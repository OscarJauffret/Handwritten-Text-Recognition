import torch
from src.config import Config


def encode_texts(char_to_idx, texts):
    """
    Convert a list of strings to a tensor containing the encoded indices.

    :param texts: List of strings
    :return: Tensor containing the encoded indices
    """
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


def decode_output(char_to_idx, output):
    """
    Decode the output of the model to text.

    :param output: Output of the model
    :return: Decoded text
    """
    _, indices = torch.max(output, 2) # Get the index with the highest probability
    indices = indices.squeeze(1).cpu().numpy()  # Remove the batch dimension and convert to NumPy

    decoded_text = ""
    prev_idx = None

    for idx in indices:
        if idx != 0 and idx != prev_idx:  # Ignore the blank character and consecutive duplicates
            decoded_text += list(char_to_idx.keys())[list(char_to_idx.values()).index(idx)]  # Convert the index to the corresponding character
        prev_idx = idx

    return decoded_text