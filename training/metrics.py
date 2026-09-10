"""
Metrics computation module for evaluation metrics: Accuracy, Precision, Recall, F1 Score.
"""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "precision_weighted": float(weighted_prec),
        "recall_weighted": float(weighted_rec),
        "f1_weighted": float(weighted_f1),
        "classification_report_dict": report_dict,
        "classification_report_str": report_str,
    }
