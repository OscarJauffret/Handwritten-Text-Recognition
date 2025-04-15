import torch
import torch.optim as optim

from src.model.CRNN import CRNN
import torch.nn as nn

from .utils.utils import encode_texts
from src.model.dataset import HandwritingDataset
from torch.utils.data import DataLoader
from config import Config

def train():
    torch.backends.cudnn.benchmark = True
    image_folder = Config.Paths.train_words
    label_folder = Config.Paths.train_labels
    dataset = HandwritingDataset(image_folder, label_folder)

    # Training a model on a single image at a time would be too slow, so we create a batch of 8 images which will be fed to the model at once.
    # A batch is a mini-group that will do a single update of the model's weights.
    # A batch size of 8 is a good compromise between speed and not too much memory usage.
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)  # Shuffling avoids the model learning the order of the images

    val_image_folder = Config.Paths.validate_words
    val_label_folder = Config.Paths.validate_labels
    val_dataset = HandwritingDataset(val_image_folder, val_label_folder)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # This is the alphabet that the model will learn to recognize
    # All the characters that can be recognized. This is the vocabulary of the model, it only recognizes these characters
    # Associate each character with an index. We add 1 because the CTC loss expects the blank character to be at index 0
    char_to_idx = {char: idx + 1 for idx, char in enumerate(Config.Model.alphabet)}
    char_to_idx["<BLANK>"] = 0  # Blank character for CTC loss

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the model, loss function, and optimizer
    num_classes = len(Config.Model.alphabet) + 1  # +1 for the blank character
    crnn = CRNN(num_classes=num_classes).to(device)

    # CTC loss uses BLANK=0 to align the predictions without knowing the exact position of the characters
    # For example, if  the model predicts "HHHEELLLOO", the CTC loss will align it to "HELLO" by removing the BLANKS
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) # CTC loss doesn't require to have a text that is cut letter by letter unlike the cross-entropy loss

    # The optimizer automatically updates the learning rate of the model. It works well with CRNNs
    optimizer = optim.Adam(crnn.parameters(), lr=0.0001)

    torch.autograd.set_detect_anomaly(True) # Will halt training and point to the first place where something goes wrong

    best_val_loss = float("inf")
    patience = 5  # nombre d'époques sans amélioration avant arrêt
    patience_counter = 0

    # Training
    for epoch in range(Config.Model.epochs):
        print(f"Epoch {epoch+1}/{Config.Model.epochs}")
        crnn.train()    # Set the model to training mode
        total_loss = 0  # Accumulate the loss for each epoch

        for images, texts in train_loader:  #  Images is a tensor of shape (batch_size, 1, height, width)
            images = images.to(device)

            targets = encode_texts(char_to_idx, texts)  # Convert the texts to indices

            # The CTC loss requires the number of columns that the model outputs for each image
            input_lengths = torch.full((images.shape[0],), Config.Model.output_width, dtype=torch.long).to(device)

            # CTC loss uses the length of the target text to align the model output with the real texts
            # For example, if the texts are ["HELLO", "WORLD"], the target_lengths will be [5, 5]
            target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long).to(device)

            # Forward pass
            outputs = crnn(images)  # Pass the images through the model
            outputs = outputs.log_softmax(2)  # For the CTC loss, we need to apply the log_softmax to the output. CTC needs log-normalized probabilities
            # outputs is the predictions of the model. It is a tensor of shape (batch_size, 32, num_classes). Each class has a probability for each of the 32 columns

            if torch.isnan(outputs).any():  # https://discuss.pytorch.org/t/best-practices-to-solve-nan-ctc-loss/151913
                print(f"{Config.Colors.error}⚠️ NaN detected in the output. Skipping this batch.{Config.Colors.reset}", end=" ")
                print(f"{Config.Colors.gray}Texts: {texts}{Config.Colors.reset}")
                continue


            # Calculate the loss
            # targets is the real text
            # input_lengths is the number of columns that the model outputs for each image
            # target_lengths is the length of the real text
            loss = criterion(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            optimizer.zero_grad()   # Clear the gradients.
            loss.backward()         # Calculate the gradients
            optimizer.step()        # Update the weights

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{Config.Model.epochs}, Loss: {total_loss / len(train_loader):.4f}")


        # Validation phase
        crnn.eval()
        val_loss = 0
        with torch.no_grad():
            for images, texts in val_loader:
                images = images.to(device)
                targets = encode_texts(char_to_idx, texts)
                input_lengths = torch.full((images.shape[0],), Config.Model.output_width, dtype=torch.long).to(device)
                target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long).to(device)
                outputs = crnn(images).log_softmax(2)
                loss = criterion(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the model after each epoch
        torch.save(crnn.state_dict(), 'model_words.pth')


if __name__ == '__main__':
    train()