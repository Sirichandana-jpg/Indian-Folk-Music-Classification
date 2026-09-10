"""
Base PyTorch Model class for Assamese Folk Music Deep Learning models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseFolkMusicModel(nn.Module, ABC):
    """Abstract Base Class for all Assam Folk Music deep learning models."""

    def __init__(self, num_classes: int = 10):
        super(BaseFolkMusicModel, self).__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking input spectrogram tensor (B, C, H, W) -> (B, num_classes)."""
        pass

    def get_num_params(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
