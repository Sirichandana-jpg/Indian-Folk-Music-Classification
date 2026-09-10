"""
CNN + BiLSTM hybrid architecture for learning local spectral features and global temporal dynamics.
"""

import torch
import torch.nn as nn
from models.base_model import BaseFolkMusicModel


class CNNBiLSTM(BaseFolkMusicModel):
    """Hybrid CNN + Bidirectional LSTM model for music classification."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, hidden_dim: int = 128, lstm_layers: int = 2):
        super(CNNBiLSTM, self).__init__(num_classes=num_classes)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Pool frequen cy, preserve time

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, None)), # Reduce frequency dimension to 16
        )

        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channel, Freq, Time)
        conv_out = self.conv_block(x) # (B, C, F_sub, T)
        b, c, f, t = conv_out.shape

        # Reshape to (Batch, Time, Features) for LSTM
        seq_in = conv_out.permute(0, 3, 1, 2).reshape(b, t, c * f)

        lstm_out, _ = self.lstm(seq_in) # (B, T, hidden_dim * 2)

        # Global average pool over time dimension
        out = torch.mean(lstm_out, dim=1) # (B, hidden_dim * 2)
        logits = self.fc(out)
        return logits
