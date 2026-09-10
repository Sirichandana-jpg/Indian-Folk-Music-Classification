"""
Convolutional Recurrent Neural Network (CRNN) tailored for music signal recognition.
"""

import torch
import torch.nn as nn
from models.base_model import BaseFolkMusicModel


class CRNNModel(BaseFolkMusicModel):
    """CRNN architecture combining deep CNN layers with GRU/LSTM recurrent layers."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, rnn_hidden: int = 128):
        super(CRNNModel, self).__init__(num_classes=num_classes)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((8, None)),
        )

        self.rnn = nn.GRU(
            input_size=256 * 8,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4,
        )

        self.dense = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 128),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        features = self.cnn(x) # (B, C_out, F_out, T_out)
        b, c, f, t = features.shape

        # Permute for GRU sequence: (B, T_out, C_out * F_out)
        seq = features.permute(0, 3, 1, 2).reshape(b, t, c * f)

        rnn_out, _ = self.rnn(seq)
        # Use final time step representation
        pooled = torch.mean(rnn_out, dim=1)
        logits = self.dense(pooled)
        return logits
