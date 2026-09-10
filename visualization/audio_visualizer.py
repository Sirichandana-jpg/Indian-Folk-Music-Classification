"""
Audio & Model Visualizer module for rendering high-quality Matplotlib & Seaborn plots.
Generates Waveform, Mel-Spectrogram, MFCC, Chroma, Confusion Matrix, ROC curves, and Training curves.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

from utils.logger import setup_logger

logger = setup_logger(__name__)
sns.set_theme(style="darkgrid")


class AudioVisualizer:
    """Renders acoustic plots, signal graphs, and training/evaluation figures."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plots_dir = Path(config["outputs"]["plots_dir"])
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.sr = config["dataset"]["target_sr"]

    def plot_waveform(self, y: np.ndarray, title: str = "Audio Waveform", save_name: Optional[str] = None) -> plt.Figure:
        """Plots time-domain audio waveform."""
        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
        librosa.display.waveshow(y, sr=self.sr, color="#1f77b4", ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel("Amplitude", fontsize=11)
        plt.tight_layout()

        if save_name:
            plt.savefig(self.plots_dir / f"{save_name}.png")
        return fig

    def plot_mel_spectrogram(self, mel_spec_db: np.ndarray, title: str = "Mel Spectrogram (dB)", save_name: Optional[str] = None) -> plt.Figure:
        """Plots Mel-Spectrogram heatmap."""
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        img = librosa.display.specshow(
            mel_spec_db, sr=self.sr, hop_length=self.config["features"]["hop_length"],
            x_axis="time", y_axis="mel", fmax=self.config["features"]["fmax"], cmap="magma", ax=ax
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        plt.tight_layout()

        if save_name:
            plt.savefig(self.plots_dir / f"{save_name}.png")
        return fig

    def plot_mfcc(self, mfcc: np.ndarray, title: str = "MFCC Coefficients", save_name: Optional[str] = None) -> plt.Figure:
        """Plots 40 MFCC coefficients heatmap."""
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        img = librosa.display.specshow(
            mfcc, sr=self.sr, hop_length=self.config["features"]["hop_length"], x_axis="time", cmap="coolwarm", ax=ax
        )
        fig.colorbar(img, ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel("MFCC Coefficients", fontsize=11)
        plt.tight_layout()

        if save_name:
            plt.savefig(self.plots_dir / f"{save_name}.png")
        return fig

    def plot_chroma(self, chroma: np.ndarray, title: str = "Chroma Pitch Features", save_name: Optional[str] = None) -> plt.Figure:
        """Plots Chroma pitch class heatmap."""
        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
        img = librosa.display.specshow(
            chroma, sr=self.sr, hop_length=self.config["features"]["hop_length"], x_axis="time", y_axis="chroma", cmap="viridis", ax=ax
        )
        fig.colorbar(img, ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        plt.tight_layout()

        if save_name:
            plt.savefig(self.plots_dir / f"{save_name}.png")
        return fig

    def plot_confusion_matrix(self, cm: np.ndarray, genres: List[str], model_name: str) -> plt.Figure:
        """Plots normalized confusion matrix heatmap."""
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=genres, yticklabels=genres, ax=ax, cbar=True
        )
        ax.set_title(f"Normalized Confusion Matrix - {model_name}", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Genre", fontsize=12, labelpad=8)
        ax.set_ylabel("True Genre", fontsize=12, labelpad=8)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"confusion_matrix_{model_name}.png")
        return fig

    def plot_training_curves(self, history: Dict[str, List[float]], model_name: str) -> plt.Figure:
        """Plots training and validation loss/F1 curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss Curve
        ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
        ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
        ax1.set_title(f"Training & Validation Loss ({model_name})", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
        ax1.legend(fontsize=10)

        # F1 Curve
        ax2.plot(epochs, history["train_f1"], "o-", label="Train F1", color="#2ca02c", linewidth=2)
        ax2.plot(epochs, history["val_f1"], "s--", label="Val F1", color="#d62728", linewidth=2)
        ax2.set_title(f"Macro F1-Score Curve ({model_name})", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("F1 Score", fontsize=11)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(self.plots_dir / f"training_curves_{model_name}.png")
        return fig

    def plot_roc_curves(self, roc_data: Dict[str, Any], model_name: str) -> plt.Figure:
        """Plots Multi-class ROC curves."""
        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
        for genre, data in roc_data.items():
            ax.plot(data["fpr"], data["tpr"], label=f"{genre} (AUC = {data['auc']:.2f})", linewidth=1.8)
        ax.plot([0, 1], [0, 1], "k--", label="Random Chance", linewidth=1)
        ax.set_title(f"Multi-class ROC Curves - {model_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"roc_curves_{model_name}.png")
        return fig
