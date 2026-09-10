import pandas as pd

from utils.config_parser import load_config
from models.model_factory import build_model
from training.trainer import Trainer


# =========================
# Load configuration
# =========================
config = load_config("configs/config.yaml")


# =========================
# Load metadata
# =========================
train_df = pd.read_csv(
    "dataset/indian_folk_31/metadata/train_metadata.csv"
)

val_df = pd.read_csv(
    "dataset/indian_folk_31/metadata/val_metadata.csv"
)

test_df = pd.read_csv(
    "dataset/indian_folk_31/metadata/test_metadata.csv"
)


# =========================
# Build CNN-BiLSTM model
# =========================
model = build_model("cnn_bilstm", config)


# =========================
# Create trainer
# =========================
trainer = Trainer(
    model=model,
    model_name="cnn_bilstm",
    config=config,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df
)


# =========================
# Load best checkpoint
# =========================
trainer.load_best_checkpoint()


# =========================
# Evaluate on TEST set
# =========================
metrics = trainer.evaluate_test()


# =========================
# Print results
# =========================
print("\n" + "=" * 60)
print("CNN-BiLSTM TEST RESULTS")
print("=" * 60)

print(f"Test Accuracy       : {metrics['accuracy']:.4f}")
print(f"Macro Precision     : {metrics['precision_macro']:.4f}")
print(f"Macro Recall        : {metrics['recall_macro']:.4f}")
print(f"Macro F1            : {metrics['f1_macro']:.4f}")
print(f"Weighted Precision  : {metrics['precision_weighted']:.4f}")
print(f"Weighted Recall     : {metrics['recall_weighted']:.4f}")
print(f"Weighted F1        : {metrics['f1_weighted']:.4f}")

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(metrics["classification_report_str"])

print("=" * 60)