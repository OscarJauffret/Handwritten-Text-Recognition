from io import text_encoding

import torch
import torch.optim as optim

from CRNN import CRNN
import torch.nn as nn
from dataset import HandwritingDataset
import string
from torch.utils.data import DataLoader
from config import Config

def train():
    torch.backends.cudnn.benchmark = True


    image_folder = Config.Paths.train_lines
    label_folder = Config.Paths.train_labels
    dataset = HandwritingDataset(image_folder, label_folder)

    # Training a model on a single image at a time would be too slow, so we create a batch of 8 images which will be fed to the model at once.
    # A batch is a mini-group that will do a single update of the model's weights.
    # A batch size of 8 is a good compromise between speed and not too much memory usage.
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)  # Shuffling avoids the model learning the order of the images

    # This is the alphabet that the model will learn to recognize
    # All the characters that can be recognized. This is the vocabulary of the model, it only recognizes these characters
    ALPHABET = string.ascii_lowercase + string.ascii_uppercase + string.digits + " .,;'-()\"!&:?#+*/"
    # Associate each character with an index. We add 1 because the CTC loss expects the blank character to be at index 0
    char_to_idx = {char: idx + 1 for idx, char in enumerate(ALPHABET)}
    char_to_idx["<BLANK>"] = 0  # Blank character for CTC loss


    def encode_texts(texts):
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

    def decode_output(output):
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

    # Hyperparameters
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model, loss function, and optimizer
    num_classes = len(ALPHABET) + 1  # +1 for the blank character
    crnn = CRNN(num_classes=num_classes).to(device)

    # CTC loss uses BLANK=0 to align the predictions without knowing the exact position of the characters
    # For example, if  the model predicts "HHHEELLLOO", the CTC loss will align it to "HELLO" by removing the BLANKS
    criterion = nn.CTCLoss(blank=0) # CTC loss doesn't require to have a text that is cut letter by letter unlike the cross-entropy loss

    # The optimizer automatically updates the learning rate of the model. It works well with CRNNs
    optimizer = optim.Adam(crnn.parameters(), lr=0.0005)

    # Training
    for epoch in range(num_epochs):
        crnn.train()    # Set the model to training mode
        total_loss = 0  # Accumulate the loss for each epoch

        for images, texts in train_loader:  #  Images is a tensor of shape (batch_size, 1, height, width)
            images = images.to(device)

            targets = encode_texts(texts)  # Convert the texts to indices

            # The CTC loss requires the number of columns that the model outputs for each image
            input_lengths = torch.full((images.shape[0],), Config.Model.output_width, dtype=torch.long).to(device)

            # CTC loss uses the length of the target text to align the model output with the real texts
            # For example, if the texts are ["HELLO", "WORLD"], the target_lengths will be [5, 5]
            target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long).to(device)

            # Forward pass
            outputs = crnn(images)  # Pass the images through the model
            outputs = outputs.log_softmax(2)  # For the CTC loss, we need to apply the log_softmax to the output. CTC needs log-normalized probabilities
            # outputs is the predictions of the model. It is a tensor of shape (batch_size, 32, num_classes). Each class has a probability for each of the 32 columns

            if torch.isnan(outputs).any():
                print(f"{Config.Colors.error}NaN in the output{Config.Colors.reset}")
                exit()


            # Calculate the loss
            # targets is the real text
            # input_lengths is the number of columns that the model outputs for each image
            # target_lengths is the length of the real text
            loss = criterion(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            optimizer.zero_grad()   # Clear the gradients. If we don't do this, the gradients will accumulate for each batch
            loss.backward()         # Calculate the gradients
            optimizer.step()        # Update the weights

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
        # Save the model after each epoch
        torch.save(crnn.state_dict(), 'model.pth')


if __name__ == '__main__':
    train()