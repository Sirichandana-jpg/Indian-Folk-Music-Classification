# Indian Folk Music Classification Using Deep Learning

Deep learning based system for classifying Indian folk music using audio
signal processing, feature extraction, CNN, BiLSTM and CRNN models.

## Target Categories

- Bihu
- Kamrupi_Lokgeet
- Yakshagana
- Rouf
- Lavani
- Uttarakhandi

## Dataset

- 1,807 audio clips
- 299 identifiable source songs
- Source-level split: 70:15:15
- Train: 1,267 clips
- Validation: 270 clips
- Test: 270 clips

## Features

- Mel-Spectrogram
- MFCC
- Chroma
- Input shape: `(3, 128, 216)`

## Data Augmentation

The final experiment uses feature-domain SpecAugment with frequency
masking and time masking.

## Models

- 2D CNN
- CNN + BiLSTM
- CRNN
- CRNN + SpecAugment

## Best Result

- Model: CRNN + SpecAugment
- Test Accuracy: 56.30%
- Test Macro F1: 55.77%

## Project Structure

```text
Indian-Folk-Music-Classification/
├── augmentation/
├── configs/
├── dataset/
│   └── indian_folk_31/
│       └── metadata/
│           ├── class_counts.csv
│           ├── train_metadata.csv
│           ├── val_metadata.csv
│           └── test_metadata.csv
├── feature_extraction/
├── models/
│   ├── base_model.py
│   ├── cnn_2d.py
│   ├── cnn_bilstm.py
│   ├── crnn.py
│   └── model_factory.py
├── preprocessing/
├── saved_models/
│   ├── best_cnn.pt
│   ├── best_cnn_bilstm.pt
│   ├── best_crnn.pt
│   └── best_crnn_specaugment.pt
├── streamlit_app/
│   ├── app.py
│   └── app_utils.py
├── training/
├── utils/
├── visualization/
├── build_metadata.py
├── split_dataset.py
├── extract_dataset_features.py
├── train.py
├── predict.py
├── evaluate_cnn.py
├── evaluate_cnn_bilstm.py
├── evaluate_crnn.py
├── requirements.txt
└── README.md
```

## Run the Application
git clone https://github.com/Sirichandana-jpg/Indian-Folk-Music-Classification.git
cd Indian-Folk-Music-Classification
pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
python -m streamlit run streamlit_app/app.py

## Project Report


## Internship

- Program: Indian Knowledge System Internship Program 2026
- Intern: Jalagam Sirichandana
- Mentor: Dr. Ripon Patgiri
- Principal Investigator: Dr. Anupam Biswas
- Institute: National Institute of Technology Silchar
- Duration: 22/06/2026 - 21/08/2026

## Limitations

The dataset is limited, class distributions are unequal, and some
categories have overlapping acoustic characteristics. The labels should
be considered dataset-level experimental labels rather than expert-
validated musicological ground truth.

## Future Work

Future work includes expert label validation, larger datasets, raw-audio
augmentation, longer temporal context, pretrained audio models,
transformers, explainability, and cross-dataset evaluation.
