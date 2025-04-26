import torch
import os
import datetime
from torch import optim
import torch.nn as nn
import editdistance
from ..config import Config
from ..utils.utils import encode_texts, decode_output

class Trainer:
    def __init__(self, model, lr, patience, device):
        self.model = model

        # The optimizer automatically updates the learning rate of the models. It works well with CRNNs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

        # CTC loss uses BLANK=0 to align the predictions without knowing the exact position of the characters
        # For example, if  the models predicts "HHHEELLLOO", the CTC loss will align it to "HELLO" by removing the BLANKS
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # CTC loss doesn't require to have a text that is cut letter by letter unlike the cross-entropy loss
        # Warning: If running on MPS, CTC loss is not supported. You should fall back to CPU for the loss calculation.
        # Use the following command in the terminal to fall back to CPU: export PYTORCH_ENABLE_MPS_FALLBACK=1
        self.patience = patience  # Number of epochs without improvement before stopping
        self.patience_counter = 0
        self.best_val_cer = float("inf")  # Initialize the best validation character error rate

    def train_step(self, images, texts):
        input_lengths, outputs, target_lengths, targets = self.forward(images, texts)

        if torch.isnan(outputs).any():  # https://discuss.pytorch.org/t/best-practices-to-solve-nan-ctc-loss/151913
            print(f"{Config.Colors.error}⚠️ NaN detected in the output. Skipping this batch.{Config.Colors.reset}", end=" ")
            print(f"{Config.Colors.gray}Texts: {texts}{Config.Colors.reset}")
            return

        # Calculate the loss
        # targets is the real text
        # input_lengths is the number of columns that the models outputs for each image
        # target_lengths is the length of the real text
        # Permute the output because we want the batch dimension to be the first one
        loss = self.criterion(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)
        self.optimizer.zero_grad()  # Clear the gradients.
        loss.backward()  # Calculate the gradients
        self.optimizer.step()  # Update the weights

        return loss.item()

    def forward(self, images, texts):
        images = images.to(self.device)
        targets = encode_texts(texts).to(self.device)  # Convert the texts to indices
        # The CTC loss requires the number of columns that the models outputs for each image
        input_lengths = torch.full((images.shape[0],), Config.Model.output_width, dtype=torch.long).to(self.device)
        # CTC loss uses the length of the target text to align the models output with the real texts
        # For example, if the texts are ["HELLO", "WORLD"], the target_lengths will be [5, 5]
        target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long).to(self.device)
        # Forward pass
        outputs = self.model(images)  # Pass the images through the models
        outputs = outputs.log_softmax(2)  # For the CTC loss, we need to apply the log_softmax to the output. CTC needs log-normalized probabilities
        # outputs is the predictions of the models. It is a tensor of shape (batch_size, 32, num_classes). Each class has a probability for each of the 32 columns
        return input_lengths, outputs, target_lengths, targets

    def validate_step(self, images, texts):
        """
        Validates the model's step by computing the Character Error Rate (CER) of predictions
        against ground truth texts

        :param images: A batch of input image tensors representing textual information.
        :param texts: A list of ground truth strings associated with the input images.
        :return: The average Character Error Rate (CER) computed across all predictions in the batch.
        """
        input_lengths, outputs, target_lengths, targets = self.forward(images, texts)
        predictions = decode_output(outputs)

        total_cer = 0
        for pred_text, true_text in zip(predictions, texts):
            cer = editdistance.eval(pred_text, true_text) / max(len(true_text), 1)
            total_cer += cer

        average_cer = total_cer / len(texts)
        return average_cer

    def train(self, train_loader, val_loader, epochs):
        if not os.path.exists(Config.Paths.models_path):
            os.makedirs(Config.Paths.models_path)
        print(f"Training for {'∞' if epochs == -1 else epochs} epochs...")
        epoch = 0
        while True:
            if epochs != -1 and epoch >= epochs:
                break
            self.model.train()
            epoch_loss = 0
            for images, texts in train_loader:
                loss = self.train_step(images, texts)
                if loss is not None:
                    epoch_loss += loss
            epoch_loss /= len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs if epochs != -1 else '∞'}, Loss: {epoch_loss:.4f}")

            # Validation phase
            val_cer = 0
            self.model.eval()
            with torch.no_grad():
                for images, texts in val_loader:
                    cer = self.validate_step(images, texts)
                    val_cer += cer
            val_cer /= len(val_loader)
            print(f"Validation CER: {val_cer:.4f}")

            if val_cer < self.best_val_cer:
                self.best_val_cer = val_cer
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(Config.Paths.models_path, "best_model.pth"))
                print("New best model saved.")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            # Save the models after each epoch
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(self.model.state_dict(), os.path.join(Config.Paths.models_path, f"model_{timestamp}.pth"))
            epoch += 1