from pathlib import Path
import pandas as pd
import re
import random

ROOT = Path(
    "dataset/indian_folk_31/raw_sources/"
    "Indian-Folk-Songs/audio chunks"
)

OUT = Path("dataset/indian_folk_31/metadata")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

rows = []

# ---------------------------------------------------------
# Collect source songs first.
# All chunks belonging to one source song stay together.
# ---------------------------------------------------------

source_groups = {}

for class_dir in ROOT.iterdir():

    if not class_dir.is_dir():
        continue

    source_language = class_dir.name

    for wav in class_dir.glob("*.wav"):

        match = re.search(
            r"(\d+)chunk\((\d+)\)",
            wav.stem,
            re.IGNORECASE
        )

        if match:
            source_number = match.group(1)
            chunk_number = match.group(2)
            source_id = f"{source_language}_{source_number}"
        else:
            source_id = wav.stem
            chunk_number = ""

        source_groups.setdefault(
            source_language, {}
        ).setdefault(source_id, []).append(
            (wav, chunk_number)
        )

# ---------------------------------------------------------
# Requested mapping
# ---------------------------------------------------------

label_mapping = {
    "Kannada": "Yakshagana",
    "Kashmiri": "Rouf",
    "Marathi": "Lavani",
    "Uttarakhandi": "Uttarakhandi",
}

# ---------------------------------------------------------
# Assamese:
# split SOURCE SONGS 50/50 between Bihu and Kamrupi_Lokgeet
# ---------------------------------------------------------

assamese_sources = list(
    source_groups.get("Assamese", {}).keys()
)

random.shuffle(assamese_sources)

mid = len(assamese_sources) // 2

assamese_label = {}

for source_id in assamese_sources[:mid]:
    assamese_label[source_id] = "Bihu"

for source_id in assamese_sources[mid:]:
    assamese_label[source_id] = "Kamrupi_Lokgeet"

# ---------------------------------------------------------
# Build metadata
# ---------------------------------------------------------

for language, sources in source_groups.items():

    for source_id, files in sources.items():

        if language == "Assamese":
            class_name = assamese_label[source_id]

        elif language in label_mapping:
            class_name = label_mapping[language]

        else:
            continue

        for wav, chunk_number in files:

            rows.append({
                "file_path": str(wav),
                "filename": wav.name,
                "class_name": class_name,
                "source_language": language,
                "source_id": source_id,
                "chunk_id": chunk_number,
                "source_dataset":
                    "A-dataset-of-Indian-Folk-Songs",
                "labeling_method":
                    "source_category_proxy_mapping",
                "verification_status":
                    "proxy_label_not_manual_style_verification"
            })

df = pd.DataFrame(rows)

# ---------------------------------------------------------
# Assign class IDs
# ---------------------------------------------------------

classes = [
    "Bihu",
    "Kamrupi_Lokgeet",
    "Yakshagana",
    "Rouf",
    "Lavani",
    "Uttarakhandi",
]

class_map = {
    name: i
    for i, name in enumerate(classes)
}

df["class_id"] = df["class_name"].map(class_map)

df = df.sort_values(
    ["class_id", "source_id", "chunk_id"]
).reset_index(drop=True)

# ---------------------------------------------------------
# Save metadata
# ---------------------------------------------------------

master_path = OUT / "master_metadata.csv"

df.to_csv(
    master_path,
    index=False
)

# Class counts
counts = (
    df.groupby(
        ["class_id", "class_name"]
    )
    .agg(
        clips=("filename", "count"),
        source_songs=("source_id", "nunique")
    )
    .reset_index()
)

counts.to_csv(
    OUT / "class_counts.csv",
    index=False
)

# ---------------------------------------------------------
# Print report
# ---------------------------------------------------------

print("=" * 70)
print("INDIAN FOLK MUSIC DATASET METADATA")
print("=" * 70)

print(f"Total clips : {len(df)}")
print(
    f"Total source songs : "
    f"{df['source_id'].nunique()}"
)

print()
print("CLASS DISTRIBUTION")
print("-" * 70)
print(counts.to_string(index=False))

print()
print("SOURCE-LANGUAGE → CLASS")
print("-" * 70)

for language in source_groups:
    if language == "Assamese":
        print(
            "Assamese → Bihu + Kamrupi_Lokgeet "
            "(source-song level 50/50 split)"
        )
    elif language in label_mapping:
        print(
            f"{language} → {label_mapping[language]}"
        )

print()
print("Metadata saved:")
print(master_path)

print()
print("IMPORTANT:")
print(
    "Labels are proxy labels based on source-category "
    "mapping and are not manually verified folk-style labels."
)