from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from feature_extraction.feature_extractor import FeatureExtractor


# ============================================================
# PATHS
# ============================================================

CONFIG_PATH = Path("configs/config.yaml")

METADATA_DIR = Path(
    "dataset/indian_folk_31/metadata"
)

FEATURE_DIR = Path(
    "dataset/indian_folk_31/features"
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

extractor = FeatureExtractor(config)

TARGET_SR = config["dataset"]["target_sr"]


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):

    metadata_path = (
        METADATA_DIR /
        f"{split_name}_metadata.csv"
    )

    output_dir = FEATURE_DIR / split_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(metadata_path)

    print()
    print("=" * 70)
    print(f"EXTRACTING {split_name.upper()} FEATURES")
    print("=" * 70)

    print(f"Files to process: {len(df)}")
    print(f"Output directory: {output_dir}")

    successful = 0
    failed = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{split_name} features"
    ):

        audio_path = Path(row["file_path"])

        # Create a unique feature filename
        feature_name = (
            f"{row['class_id']}_"
            f"{row['source_id']}_"
            f"chunk_{row['chunk_id']}.npy"
        )

        output_path = output_dir / feature_name

        # Skip if already extracted
        if output_path.exists():
            successful += 1
            continue

        try:

            # ------------------------------------------------
            # LOAD AUDIO
            # ------------------------------------------------

            y, sr = librosa.load(
                audio_path,
                sr=TARGET_SR,
                mono=True
            )

            # ------------------------------------------------
            # EXTRACT STACKED FEATURES
            # ------------------------------------------------

            features = (
                extractor.extract_stacked_features_2d(y)
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            np.save(
                output_path,
                features
            )

            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"\nERROR: {audio_path}"
            )

            print(
                f"Reason: {e}"
            )

    print()
    print(f"{split_name.upper()} COMPLETE")
    print("-" * 70)
    print(f"Successful: {successful}")
    print(f"Failed    : {failed}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("INDIAN FOLK MUSIC FEATURE EXTRACTION")
    print("=" * 70)

    for split in [
        "train",
        "val",
        "test"
    ]:

        process_split(split)

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print()
    print("Features saved in:")
    print(FEATURE_DIR)