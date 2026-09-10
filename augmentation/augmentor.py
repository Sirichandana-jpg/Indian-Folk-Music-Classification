"""
Data Augmentation module for raw audio signals and spectrogram representations.
Implements Pitch Shift, Time Stretch, Volume Scaling, Gaussian Noise, Random Crop, and SpecAugment.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import librosa

from utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioAugmentor:
    """Data augmentation engine for time-domain raw audio and 2D spectrograms."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        aug_cfg = config.get("augmentation", {})
        self.sr = config["dataset"]["target_sr"]
        self.pitch_range = aug_cfg.get("pitch_shift_steps", [-2, 2])
        self.stretch_range = aug_cfg.get("time_stretch_rates", [0.8, 1.2])
        self.noise_factor = aug_cfg.get("noise_factor", 0.005)
        self.volume_gain_db = aug_cfg.get("volume_gain_db", [-6, 6])
        
        spec_cfg = aug_cfg.get("spec_augment", {})
        self.freq_mask_param = spec_cfg.get("freq_mask_param", 15)
        self.time_mask_param = spec_cfg.get("time_mask_param", 35)
        self.n_freq_masks = spec_cfg.get("n_freq_masks", 2)
        self.n_time_masks = spec_cfg.get("n_time_masks", 2)

    def pitch_shift(self, y: np.ndarray) -> np.ndarray:
        """Applies pitch shifting randomly within configured steps."""
        n_steps = np.random.uniform(self.pitch_range[0], self.pitch_range[1])
        return librosa.effects.pitch_shift(y=y, sr=self.sr, n_steps=n_steps)

    def time_stretch(self, y: np.ndarray) -> np.ndarray:
        """Applies time stretching randomly within configured rates."""
        rate = np.random.uniform(self.stretch_range[0], self.stretch_range[1])
        y_stretched = librosa.effects.time_stretch(y=y, rate=rate)
        # Re-pad or trim to maintain fixed duration length
        target_len = int(self.sr * self.config["dataset"]["segment_duration"])
        if len(y_stretched) > target_len:
            return y_stretched[:target_len]
        else:
            return np.pad(y_stretched, (0, target_len - len(y_stretched)), mode="constant")

    def add_gaussian_noise(self, y: np.ndarray) -> np.ndarray:
        """Adds random Gaussian background noise to audio signal."""
        noise = np.random.randn(len(y))
        return y + self.noise_factor * noise

    def volume_scaling(self, y: np.ndarray) -> np.ndarray:
        """Scales gain / volume level randomly."""
        gain_db = np.random.uniform(self.volume_gain_db[0], self.volume_gain_db[1])
        scale = 10 ** (gain_db / 20.0)
        return y * scale

    def random_crop(self, y: np.ndarray, crop_duration: float = 4.0) -> np.ndarray:
        """Crops a random temporal snippet of audio."""
        crop_len = int(self.sr * crop_duration)
        if len(y) <= crop_len:
            return y
        start = np.random.randint(0, len(y) - crop_len)
        cropped = y[start : start + crop_len]
        target_len = int(self.sr * self.config["dataset"]["segment_duration"])
        return np.pad(cropped, (0, target_len - crop_len), mode="constant")

    def apply_audio_augmentations(self, y: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Applies a sequence of audio domain augmentations with probability p."""
        if np.random.rand() < p:
            y = self.pitch_shift(y)
        if np.random.rand() < p:
            y = self.time_stretch(y)
        if np.random.rand() < p:
            y = self.add_gaussian_noise(y)
        if np.random.rand() < p:
            y = self.volume_scaling(y)
        return y

    def apply_spec_augment(self, spec: np.ndarray) -> np.ndarray:
        """Applies SpecAugment (Frequency and Time Masking) on 2D/3D Spectrogram numpy arrays."""
        spec_aug = spec.copy()
        
        # Determine shape (C, F, T) or (F, T)
        if spec_aug.ndim == 2:
            num_freq, num_time = spec_aug.shape
            # Frequency masking
            for _ in range(self.n_freq_masks):
                f = np.random.randint(0, min(self.freq_mask_param, num_freq))
                f0 = np.random.randint(0, num_freq - f)
                spec_aug[f0 : f0 + f, :] = 0.0

            # Time masking
            for _ in range(self.n_time_masks):
                t = np.random.randint(0, min(self.time_mask_param, num_time))
                t0 = np.random.randint(0, num_time - t)
                spec_aug[:, t0 : t0 + t] = 0.0

        elif spec_aug.ndim == 3:
            num_channels, num_freq, num_time = spec_aug.shape
            for _ in range(self.n_freq_masks):
                f = np.random.randint(0, min(self.freq_mask_param, num_freq))
                f0 = np.random.randint(0, num_freq - f)
                spec_aug[:, f0 : f0 + f, :] = 0.0

            for _ in range(self.n_time_masks):
                t = np.random.randint(0, min(self.time_mask_param, num_time))
                t0 = np.random.randint(0, num_time - t)
                spec_aug[:, :, t0 : t0 + t] = 0.0

        return spec_aug
