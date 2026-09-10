"""
Prediction CLI script for classifying an input audio file (WAV or MP3).
"""

import argparse
from pathlib import Path
import json

from utils.config_parser import load_config
from preprocessing.audio_preprocessor import AudioPreprocessor
from streamlit_app.app_utils import load_model_for_inference, predict_audio_genre
from utils.logger import setup_logger

logger = setup_logger("PredictCLI")


def predict_single_file(audio_path: str, model_name: str = "cnn", config_path: str = "configs/config.yaml"):
    """Classifies an individual audio file and prints prediction breakdown."""
    config = load_config(config_path)
    preprocessor = AudioPreprocessor(
        target_sr=config["dataset"]["target_sr"],
        mono=config["dataset"]["mono"],
        segment_duration=config["dataset"]["segment_duration"],
    )

    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    logger.info(f"Loading and preprocessing input audio file: {audio_file}")
    y_raw, _ = preprocessor.load_audio(audio_file)
    y_padded = preprocessor.pad_or_trim(y_raw)

    model = load_model_for_inference(model_name, config)
    top_genre, confidence, all_probs = predict_audio_genre(y_padded, model, config)

    result = {
        "file": str(audio_file),
        "model_used": model_name,
        "predicted_genre": top_genre,
        "confidence": round(confidence * 100.0, 2),
        "top_3_predictions": list(all_probs.items())[:3],
    }

    print("\n" + "=" * 50)
    print("      FOLK MUSIC GENRE PREDICTION RESULT")
    print("=" * 50)
    print(f" Audio File        : {result['file']}")
    print(f" Model Used        : {result['model_used'].upper()}")
    print(f" Predicted Genre   : {result['predicted_genre']}")
    print(f" Confidence Score  : {result['confidence']}%")
    print("\n Top-3 Predicted Probabilities:")
    for genre, prob in result["top_3_predictions"]:
        print(f"  - {genre:<15}: {prob * 100.0:.2f}%")
    print("=" * 50 + "\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Assamese Folk Music Genre")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav or .mp3)")
    parser.add_argument("--model", type=str, default="cnn", help="Model architecture (cnn, cnn_bilstm, crnn)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    predict_single_file(audio_path=args.audio, model_name=args.model, config_path=args.config)
