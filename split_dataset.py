from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

METADATA = Path(
    "dataset/indian_folk_31/metadata/master_metadata.csv"
)

OUT_DIR = Path(
    "dataset/indian_folk_31/metadata"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA)

print("=" * 70)
print("SOURCE-LEVEL DATASET SPLIT")
print("=" * 70)

print(f"Total clips       : {len(df)}")
print(f"Total source songs: {df['source_id'].nunique()}")

# ============================================================
# GET UNIQUE SOURCE SONGS
# ============================================================

sources = (
    df[
        ["source_id", "class_name", "class_id"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

# ============================================================
# SPLIT 70 / 15 / 15
# STRATIFIED BY CLASS
# ============================================================

train_sources, temp_sources = train_test_split(
    sources,
    test_size=0.30,
    random_state=42,
    stratify=sources["class_id"]
)

val_sources, test_sources = train_test_split(
    temp_sources,
    test_size=0.50,
    random_state=42,
    stratify=temp_sources["class_id"]
)

# ============================================================
# ASSIGN SPLITS
# ============================================================

train_ids = set(train_sources["source_id"])
val_ids = set(val_sources["source_id"])
test_ids = set(test_sources["source_id"])

df["split"] = df["source_id"].apply(
    lambda x:
        "train" if x in train_ids
        else "val" if x in val_ids
        else "test"
)

# ============================================================
# SAVE
# ============================================================

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()
test_df = df[df["split"] == "test"].copy()

train_df.to_csv(
    OUT_DIR / "train_metadata.csv",
    index=False
)

val_df.to_csv(
    OUT_DIR / "val_metadata.csv",
    index=False
)

test_df.to_csv(
    OUT_DIR / "test_metadata.csv",
    index=False
)

df.to_csv(
    OUT_DIR / "master_metadata_with_split.csv",
    index=False
)

# ============================================================
# RESULTS
# ============================================================

print()
print("SOURCE SONG SPLIT")
print("-" * 70)

print(
    f"Train source songs: "
    f"{train_sources['source_id'].nunique()}"
)

print(
    f"Val source songs  : "
    f"{val_sources['source_id'].nunique()}"
)

print(
    f"Test source songs : "
    f"{test_sources['source_id'].nunique()}"
)

print()
print("CLIP SPLIT")
print("-" * 70)

print(f"Train clips: {len(train_df)}")
print(f"Val clips  : {len(val_df)}")
print(f"Test clips : {len(test_df)}")

print()
print("CLASS DISTRIBUTION")
print("-" * 70)

distribution = (
    df.groupby(["split", "class_name"])
    .agg(
        clips=("filename", "count"),
        source_songs=("source_id", "nunique")
    )
    .reset_index()
)

print(distribution.to_string(index=False))

print()
print("Files created:")
print(OUT_DIR / "train_metadata.csv")
print(OUT_DIR / "val_metadata.csv")
print(OUT_DIR / "test_metadata.csv")
print(OUT_DIR / "master_metadata_with_split.csv")

print()
print("=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)