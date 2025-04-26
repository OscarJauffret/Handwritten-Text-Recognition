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
    Decode the output tensor from the model into readable text.

    :param output: Output tensor of shape [B, T, C] (Batch, Time, Classes)
    :return: List of decoded texts, one per batch element
    """
    _, indices = torch.max(output, 2)  # Take the index of the highest probability class
    decoded_texts = []

    for seq in indices:
        decoded_text = ""
        prev_idx = None
        for idx in seq:
            idx = idx.item()
            # In CTC decoding, repeated characters without an intervening blank are considered a single instance.
            # The model uses a special blank token (index 0) to distinguish between real repeated characters (e.g., 'll' in "hello").
            # Therefore, during decoding:
            # - Ignore repeated characters if they directly follow each other without a blank.
            # - Allow repetition if separated by a blank.
            if idx != 0 and idx != prev_idx:  # Ignore blank and repeated characters
                decoded_text += Config.Model.alphabet[idx - 1]  # Convert index to character (blank is at index 0)
            prev_idx = idx
        decoded_texts.append(decoded_text)

    return decoded_texts

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
