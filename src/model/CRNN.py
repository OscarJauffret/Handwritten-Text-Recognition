import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(CRNN, self).__init__()

        # Convolutional layers
        self.cnn = nn.Sequential(
            # in_channels is 1 because we have grayscale images
            # out_channels correspond to the number of filters that we want to apply
            # kernel_size is the size of the filter
            # stride is the step of the filter
            # padding is the number of pixels to add around the image
            # The idea is that we will apply multiple filters to the image to try to detect different features
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),  # (64, 256) -> (128, 64, 256)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Max pool keeps the maximum value in a window of 2x2
            nn.MaxPool2d((2, 2)),  # (32, 128)

            # Takes the 128 channels from the previous layer and applies 256 filters
            # Each new filtered image is a combination of the previous 128 images
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (16, 64) This is the size of the output of the CNN

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d((2, 2)),  # (8, 32)
        )
        # The output of the CNN will be of the form (batch, 256, 8, 32), where batch is the number of images in the batch.
        # Each "column" of the image will be processed by the RNN. (there are 32 columns)

        # RNN layers
        # The RNN will take the output of the CNN and process it. It will model the relation between the characters in the image.
        # input_size is the number of features in the input, which is the number of channels in the last layer of the CNN
        # hidden_size is the number of features in the hidden state
        # num_layers is the number of recurrent layers
        # bidirectional is set to True to process the sequence from left to right and right to left
        # batch_first is set to True to have the input in the form (batch, sequence, feature)
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # BiLSTM -> num_classes

    def forward(self, x):
        x = self.cnn(x)  # CNN feature extraction
        x = x.mean(dim=2)

        # An LSTM expects the input to be of the form (batch, sequence length, feature size) -> (batch, 32, 256)
        x = x.permute(0, 2, 1)  # Permute the dimensions to have the sequence length as the second dimension

        x, _ = self.rnn(x)  # LSTM layers
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)  # Fully connected

        return x
