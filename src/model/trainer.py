import torch
import os
import datetime
import torch.nn as nn
import editdistance
import math
import matplotlib.pyplot as plt

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=math.floor(self.patience/5))
        self.val_cer_history = []
        self.train_loss_history = []

    def train_step(self, images, texts):
        images = images.to(self.device)

        input_lengths, outputs, target_lengths, targets = self.forward(images, texts)

        if torch.isnan(outputs).any():
            print(f"{Config.Colors.error}⚠️ NaN detected in the output. Skipping this batch.{Config.Colors.reset}", end=" ")
            print(f"{Config.Colors.gray}Texts: {texts}{Config.Colors.reset}")
            return None

        loss = self.criterion(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
            self.train_loss_history.append(epoch_loss)

            # Validation phase
            val_cer = 0
            self.model.eval()
            with torch.no_grad():
                for images, texts in val_loader:
                    cer = self.validate_step(images, texts)
                    val_cer += cer
            val_cer /= len(val_loader)
            self.val_cer_history.append(val_cer)

            print("=" * 60)
            print(f"Epoch {epoch + 1}/{epochs if epochs != -1 else '∞'}")
            print(f"Training Loss: {epoch_loss:.4f}")
            print(f"Validation CER: {val_cer:.4f}")
            print(f"Learning Rate: {self.scheduler.optimizer.param_groups[0]['lr']:.6f}")
            print("=" * 60)

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

            self.scheduler.step(val_cer)

            # Save the models after each epoch
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(self.model.state_dict(), os.path.join(Config.Paths.models_path, f"model_{timestamp}_{val_cer:.4f}.pth"))
            epoch += 1

        # Plot CER and Loss evolution
        plt.figure(figsize=(10, 7))
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation CER', color='tab:blue')
        ax1.plot(self.val_cer_history, marker='o', color='tab:blue', label='Validation CER')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True)

        ax2 = ax1.twinx()  # Create second y-axis
        ax2.set_ylabel('Training Loss', color='tab:red')
        ax2.plot(self.train_loss_history, marker='x', color='tab:red', label='Training Loss')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        plt.title('Validation CER and Training Loss Over Epochs')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(0.5, -0.05), ncol=2)
        plt.savefig(os.path.join(Config.Paths.models_path, "cer_loss_evolution.png"))
        plt.close()