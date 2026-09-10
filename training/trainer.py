"""
PyTorch Training Engine for Folk Music Deep Learning models.
Includes Mixed Precision, Early Stopping, Model Checkpointing, TensorBoard logging, and Resume capability.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional,List
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from feature_extraction.feature_extractor import FeatureExtractor
from augmentation.augmentor import AudioAugmentor
from training.metrics import compute_metrics
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FolkMusicDataset(Dataset):
    def __init__(self, df_split, config, split):
        self.df = df_split.reset_index(drop=True)
        self.config = config
        self.split = split

        self.feature_root = Path(
            "dataset/indian_folk_31/features"
        )
        self.augmentor = AudioAugmentor(config)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feature_filename = (
            f"{int(row['class_id'])}_"
            f"{row['source_id']}_"
            f"chunk_{row['chunk_id']}.npy"
        )

        feature_path = (
            self.feature_root
            / self.split
            / feature_filename
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}"
            )

        features = np.load(feature_path).astype(np.float32)
        if self.split == "train":
            features = self.augmentor.apply_spec_augment(features)

        tensor_x = torch.from_numpy(features).float()
        label = int(row["class_id"])

        return tensor_x, label

class Trainer:
    """Trainer manager handling full model training lifecycle."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        config: Dict[str, Any],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
    ):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.genres = config["genres"]

        # Device selection
        req_device = config["project"].get("device", "cuda")
        self.device = torch.device(req_device if torch.cuda.is_available() and req_device == "cuda" else "cpu")
        self.model.to(self.device)

        # Datasets & Loaders
        train_dataset = FolkMusicDataset(train_df, config, split="train")
        val_dataset = FolkMusicDataset(val_df, config, split="val")

        batch_size = config["training"]["batch_size"]
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        if self.test_df is not None:
           test_dataset = FolkMusicDataset(
           self.test_df,
           config,
           split="test"
        )

           self.test_loader = DataLoader(
           test_dataset,
           batch_size=batch_size,
           shuffle=False,
           num_workers=0
        )
        else:
           self.test_loader = None
        # Optimization & Loss
        # ============================================================
        # CLASS-WEIGHTED LOSS FOR IMBALANCED DATA
        # ============================================================

        class_counts = (
    train_df["class_id"]
    .value_counts()
    .sort_index()
)

        num_classes = len(self.genres)

        weights = (
    len(train_df) /
    (num_classes * class_counts)
)

        weights = weights.reindex(
    range(num_classes),
    fill_value=1.0
)

        class_weights = torch.tensor(
    weights.values,
    dtype=torch.float32
).to(self.device)

        logger.info(
    f"Class weights: {class_weights.cpu().numpy()}"
)

        self.criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Scheduler
        sched_type = config["training"].get("scheduler", "reduce_on_plateau")
        if sched_type == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=3
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config["training"]["epochs"]
            )

        # Mixed Precision
        self.use_amp = config["training"].get("mixed_precision", True) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda',enabled=self.use_amp)

        # Checkpoints & TensorBoard
        self.checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        log_dir = Path(config["training"]["logs_dir"]) / model_name
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Early stopping state
        self.patience = config["training"]["early_stopping_patience"]
        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Saves checkpoint file to disk."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.model_name}.pt"
        state = {
            "epoch": epoch,
            "model_name": self.model_name,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_f1": self.best_val_f1,
        }
        torch.save(state, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / f"best_{self.model_name}.pt"
            torch.save(state, best_path)
            logger.info(f"New best checkpoint saved for '{self.model_name}' at {best_path}")

    def load_checkpoint(self) -> int:
        """Resumes training from checkpoint if available."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.model_name}.pt"
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}...")
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state["state_dict"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.best_val_f1 = state.get("best_val_f1", 0.0)
            start_epoch = state.get("epoch", 0) + 1
            logger.info(f"Resuming training from epoch {start_epoch}")
            return start_epoch
        return 0
    def load_best_checkpoint(self) -> bool:
        """Load the best validation-F1 checkpoint."""

        best_path = (
        self.checkpoint_dir /
        f"best_{self.model_name}.pt"
    )

        if not best_path.exists():
           logger.warning(
            f"Best checkpoint not found: {best_path}"
        )
           return False

        state = torch.load(
        best_path,
        map_location=self.device
    )

        self.model.load_state_dict(
        state["state_dict"]
    )

        self.best_val_f1 = state.get(
        "best_val_f1",
        self.best_val_f1
    )

        logger.info(
        f"Loaded best checkpoint: {best_path}"
    )

        return True
    def evaluate_test(self) -> Dict[str, Any]:
        """Evaluate the best model on the test dataset."""

        if self.test_loader is None:
          raise RuntimeError(
            "Test DataLoader is not available."
        )

        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():

            for x_batch, y_batch in self.test_loader:

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with torch.amp.autocast(
                 device_type=self.device.type,
                 enabled=self.use_amp
            ):
                 outputs = self.model(x_batch)

                probabilities = torch.softmax(
                outputs,
                dim=1
            )

                predictions = torch.argmax(
                outputs,
                dim=1
            )

                all_preds.extend(
                predictions.cpu().numpy()
            )

                all_labels.extend(
                y_batch.cpu().numpy()
            )

                all_probs.extend(
                probabilities.cpu().numpy()
            )

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        metrics = compute_metrics(
        y_true,
        y_pred,
        self.genres
    )

        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        metrics["y_probs"] = y_probs

        return metrics
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Runs single training epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), self.genres)
        return epoch_loss, metrics["f1_macro"]

    def validate_epoch(self) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
        """Runs single validation epoch."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(x_batch)
                    loss = self.criterion(outputs, y_batch)

                running_loss += loss.item() * x_batch.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())

        val_loss = running_loss / len(self.val_loader.dataset)
        y_true, y_pred = np.array(all_labels), np.array(all_preds)
        metrics = compute_metrics(y_true, y_pred, self.genres)
        return val_loss, metrics, y_true, y_pred

    def train(self, resume: bool = False) -> Dict[str, List[float]]:
        """Main training loop across configured epochs."""
        epochs = self.config["training"]["epochs"]
        start_epoch = self.load_checkpoint() if resume else 0

        history = {
            "train_loss": [],
            "train_f1": [],
            "val_loss": [],
            "val_f1": [],
            "val_accuracy": [],
        }

        logger.info(f"Starting training for '{self.model_name}' on device [{self.device}]...")
        start_time = time.time()

        for epoch in range(start_epoch, epochs):
            train_loss, train_f1 = self.train_epoch(epoch)
            val_loss, val_metrics, _, _ = self.validate_epoch()
            val_f1 = val_metrics["f1_macro"]
            val_acc = val_metrics["accuracy"]

            # Learning rate step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_f1)
            else:
                self.scheduler.step()

            # Record metrics
            history["train_loss"].append(train_loss)
            history["train_f1"].append(train_f1)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_accuracy"].append(val_acc)

            # Tensorboard Logging
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("F1/Train", train_f1, epoch)
            self.writer.add_scalar("F1/Val", val_f1, epoch)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch)

            logger.info(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Save checkpoint & check early stopping
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, is_best=False)

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

        elapsed = time.time() - start_time

        logger.info(
    f"Training completed for '{self.model_name}' "
    f"in {elapsed:.2f} seconds."
)

        self.writer.close()

# --------------------------------------------------------
# Load BEST validation checkpoint
# --------------------------------------------------------

        self.load_best_checkpoint()

# --------------------------------------------------------
# Evaluate BEST model on TEST set
# --------------------------------------------------------

        if self.test_loader is not None:

           test_metrics = self.evaluate_test()

           history["test_metrics"] = test_metrics

           logger.info("")
           logger.info("=" * 70)
           logger.info(
               f"TEST RESULTS — {self.model_name.upper()}"
    )
           logger.info("=" * 70)

           logger.info(
        f"Test Accuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

           logger.info(
        f"Test Macro F1 : "
        f"{test_metrics['f1_macro']:.4f}"
    )

        return history