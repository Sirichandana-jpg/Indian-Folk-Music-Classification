"""
Training pipeline for the Indian Folk Music Classification project.

Uses:
    - train_metadata.csv
    - val_metadata.csv
    - test_metadata.csv
    - pre-extracted .npy features

Models:
    - CNN
    - CNN + BiLSTM
    - CRNN
"""

import argparse
from pathlib import Path
import pandas as pd
import torch

from utils.config_parser import load_config
from utils.seed import set_seed
from utils.logger import setup_logger
from models.model_factory import build_model
from training.trainer import Trainer


logger = setup_logger("TrainPipeline")


# LOAD OUR METADATA

def load_dataset_metadata():

    metadata_dir = Path(
        "dataset/indian_folk_31/metadata"
    )

    train_path = metadata_dir / "train_metadata.csv"
    val_path = metadata_dir / "val_metadata.csv"
    test_path = metadata_dir / "test_metadata.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(
        f"Train samples: {len(train_df)}"
    )

    logger.info(
        f"Validation samples: {len(val_df)}"
    )

    logger.info(
        f"Test samples: {len(test_df)}"
    )

    return train_df, val_df, test_df


# MAIN TRAINING PIPELINE


def run_pipeline(
    config_path="configs/config.yaml",
    target_models=None,
    resume=False,
):

    # 1. Load configuration
    

    config = load_config(config_path)

    set_seed(
        config["project"]["seed"]
    )

    logger.info(
        f"Classes: {config['genres']}"
    )

    logger.info(
        f"Number of classes: {len(config['genres'])}"
    )

    # --------------------------------------------------------
    # 2. Check device
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        f"Training device: {device}"
    )

    # --------------------------------------------------------
    # 3. Load metadata
    # --------------------------------------------------------

    logger.info(
        "--- Loading pre-extracted dataset ---"
    )

    train_df, val_df, test_df = (
        load_dataset_metadata()
    )

    # --------------------------------------------------------
    # 4. Target models
    # --------------------------------------------------------

    if target_models is None:
        target_models = ["cnn"]

    logger.info(
        f"Models selected: {target_models}"
    )

    
    # 5. Train models
    

    results = {}

    for model_name in target_models:

        logger.info("")
        logger.info("=" * 70)
        logger.info(
            f"TRAINING MODEL: {model_name.upper()}"
        )
        logger.info("=" * 70)

        
        # Build model
        

        model = build_model(
            model_name,
            config
        )

        # Trainer
            

        trainer = Trainer(
            model=model,
            model_name=model_name,
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        # Train
        

        history = trainer.train(
            resume=resume
        )

        results[model_name] = history

        logger.info("")
        logger.info(
            f"Finished training {model_name.upper()}"
        )

    
    # 6. Final summary
    

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)

    for model_name, history in results.items():

        if history["val_f1"]:

            best_f1 = max(
                history["val_f1"]
            )

            best_acc = max(
                history["val_accuracy"]
            )

            logger.info(
                f"{model_name.upper()} | "
                f"Best Val F1: {best_f1:.4f} | "
                f"Best Val Accuracy: {best_acc:.4f}"
            )



# COMMAND LINE


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Indian Folk Music Classification Training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn"],
        choices=[
            "cnn",
            "cnn_bilstm",
            "crnn",
        ],
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        target_models=args.models,
        resume=args.resume,
    )