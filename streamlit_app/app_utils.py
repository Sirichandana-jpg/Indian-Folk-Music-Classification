"""
Streamlit application helper utility functions for UI rendering, styling, and model inference.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go

from feature_extraction.feature_extractor import FeatureExtractor
from preprocessing.audio_preprocessor import AudioPreprocessor
from models.model_factory import build_model
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_model_for_inference(
    model_name: str, config: Dict[str, Any]
) -> torch.nn.Module:
    """Loads trained model checkpoint for Streamlit inference."""
    
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])

    if model_name == "crnn_specaugment":
        best_path = checkpoint_dir / "best_crnn_specaugment.pt"
    else:
        best_path = checkpoint_dir / f"best_{model_name}.pt"

    model_type = "crnn" if model_name == "crnn_specaugment" else model_name

    model = build_model(model_type, config)

    if best_path.exists():
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        logger.info(f"Loaded trained weights from {best_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {best_path}")

    model.eval()
    return model

def predict_audio_genre(
    y: np.ndarray, model: torch.nn.Module, config: Dict[str, Any]
) -> Tuple[str, float, Dict[str, float]]:
    """Runs prediction on audio numpy array and returns top genre, confidence, and genre probabilities dict."""
    extractor = FeatureExtractor(config)
    stacked_2d = extractor.extract_stacked_features_2d(y) # (3, 128, T)
    tensor_x = torch.from_numpy(stacked_2d).unsqueeze(0).float() # (1, 3, 128, T)

    with torch.no_grad():
        logits = model(tensor_x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    genres = config["genres"]
    prob_dict = {genres[i]: float(probs[i]) for i in range(len(genres))}

    # Sort probabilities descending
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top_genre, top_confidence = sorted_probs[0]

    return top_genre, top_confidence, dict(sorted_probs)


def create_top3_probability_chart(sorted_probs: Dict[str, float]) -> go.Figure:
    """Creates a clean Plotly horizontal bar chart for top predicted genres."""
    top3 = list(sorted_probs.items())[:3]
    genres = [item[0] for item in reversed(top3)]
    probs = [item[1] * 100.0 for item in reversed(top3)]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=genres,
            orientation="h",
            marker=dict(
                color=["#17becf", "#9467bd", "#1f77b4"],
                line=dict(color="#ffffff", width=1),
            ),
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Top-3 Genre Probability Breakdown",
        xaxis_title="Confidence Probability (%)",
        yaxis_title="Genre Class",
        xaxis=dict(range=[0, 105]),
        margin=dict(l=20, r=40, t=40, b=20),
        height=280,
    )
    return fig
