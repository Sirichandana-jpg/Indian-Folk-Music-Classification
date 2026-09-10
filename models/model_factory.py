"""
Model Factory for instantiating deep learning models dynamically.
"""

from typing import Dict, Any
import torch.nn as nn

from models.cnn_2d import Custom2DCNN
from models.cnn_bilstm import CNNBiLSTM
from models.crnn import CRNNModel
from utils.logger import setup_logger

logger = setup_logger(__name__)


def build_model(model_name: str, config: Dict[str, Any], in_channels: int = 3) -> nn.Module:
    """Factory function to build requested PyTorch model architecture."""
    name = model_name.lower().strip()
    num_classes = len(config["genres"])

    if name == "cnn":
        model = Custom2DCNN(in_channels=in_channels, num_classes=num_classes)
    elif name in ["cnn_bilstm", "cnn-bilstm", "bilstm"]:
        model = CNNBiLSTM(in_channels=in_channels, num_classes=num_classes)
    elif name in ["crnn","crnn_specaugment"]:
        model = CRNNModel(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Choose from ['cnn', 'cnn_bilstm', 'crnn','crnn_specaugment']."
        )

    logger.info(
        f"Successfully built model '{name}' with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters."
    )
    return model
