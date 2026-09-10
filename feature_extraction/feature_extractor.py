"""
Feature Extractor module for audio signal processing.
Extracts 13 distinct acoustic & spectral features using Librosa and packages them for PyTorch modeling.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Union
import numpy as np
import librosa

from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureExtractor:
    """Extracts 13 acoustic and spectral feature groups from audio signals."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sr = config["dataset"]["target_sr"]
        self.n_mfcc = config["features"]["n_mfcc"]
        self.n_mels = config["features"]["n_mels"]
        self.n_fft = config["features"]["n_fft"]
        self.hop_length = config["features"]["hop_length"]
        self.win_length = config["features"]["win_length"]
        self.fmin = config["features"]["fmin"]
        self.fmax = config["features"]["fmax"]

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extracts Mel-Spectrogram (dB scaled)."""
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def extract_all_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Extracts all 13 acoustic features from an audio array."""
        # 1. MFCC (40 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)

        # 2. Mel Spectrogram
        mel_spec_db = self.extract_mel_spectrogram(y)

        # 3. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)

        # 4. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)

        # 5. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)

        # 6. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)

        # 7. Spectral Roll-off
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)

        # 8. Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)

        # 9. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)

        # 10. Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=self.sr)

        # 11 & 12. Harmonic & Percussive Features
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_rms = librosa.feature.rms(y=y_harmonic, hop_length=self.hop_length)
        percussive_rms = librosa.feature.rms(y=y_percussive, hop_length=self.hop_length)

        # 13. Tonnetz Features
        try:
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=self.sr)
        except Exception:
            tonnetz = np.zeros((6, chroma.shape[1]))

        return {
            "mfcc": mfcc,
            "mel_spectrogram": mel_spec_db,
            "chroma": chroma,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_contrast": contrast,
            "spectral_rolloff": rolloff,
            "zcr": zcr,
            "rms": rms,
            "tempo": tempo,
            "harmonic_rms": harmonic_rms,
            "percussive_rms": percussive_rms,
            "tonnetz": tonnetz,
        }
    def extract_stacked_features_2d(self, y: np.ndarray) -> np.ndarray:
        """Extracts and stacks key 2D spectral features into a multi-channel image tensor format (C, H, W)."""
        mel = self.extract_mel_spectrogram(y)  # (128, T)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=40, n_fft=self.n_fft, hop_length=self.hop_length) # (40, T)
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length) # (12, T)
        # 40 * 4 = 160 rows -> slice to exactly 128
        mfcc_resized = np.tile(mfcc, (4, 1))[:128, :] 
    
        # 12 * 11 = 132 rows -> slice to exactly 128
        chroma_resized = np.tile(chroma, (11, 1))[:128, :] 

        # Ensure all three channels match time dimensions (T) perfectly
        min_t = min(mel.shape[1], mfcc_resized.shape[1], chroma_resized.shape[1])
        mel = mel[:, :min_t]
        mfcc_resized = mfcc_resized[:, :min_t]
        chroma_resized = chroma_resized[:, :min_t]

        # Stack into 3 channels: Channel 0 = Mel, Channel 1 = MFCC, Channel 2 = Chroma
        stacked = np.stack([mel, mfcc_resized, chroma_resized], axis=0) # (3, 128, min_t)
        return stacked.astype(np.float32)
