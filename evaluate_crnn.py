import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from models.model_factory import build_model
from training.trainer import FolkMusicDataset


# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

genres = config["genres"]
num_classes = len(genres)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Load test metadata
# ---------------------------------------------------------

test_df = pd.read_csv(
    "dataset/indian_folk_31/metadata/test_metadata.csv"
)

test_dataset = FolkMusicDataset(
    test_df,
    config,
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=0,
)


# ---------------------------------------------------------
# Build CRNN
# ---------------------------------------------------------

model = build_model(
    model_name="crnn",
    config=config
)

model = model.to(device)


# ---------------------------------------------------------
# Load best CRNN checkpoint
# ---------------------------------------------------------

checkpoint_path = Path(
    "saved_models/best_crnn.pt"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for x_batch, y_batch in test_loader:

        x_batch = x_batch.to(device)

        outputs = model(x_batch)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            y_batch.numpy()
        )


y_true = np.array(all_labels)
y_pred = np.array(all_predictions)


# ---------------------------------------------------------
# Overall results
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n" + "=" * 70)
print("CRNN TEST EVALUATION")
print("=" * 70)

print(f"\nTest Accuracy : {accuracy:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=genres,
        digits=4,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes))
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=genres,
    columns=genres
))