"""
Audio Preprocessing module for Folk Music Classification.
Provides robust signal processing functions including resampling, mono conversion,
silence removal, noise reduction, loudness normalization, fixed segmentation, padding, and trimming.
"""
from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
import librosa
import soundfile as sf
from utils.logger import setup_logger
logger = setup_logger(__name__)
class AudioPreprocessor:
    """Audio preprocessing engine for cleaning, normalizing, and segmenting audio files."""
    def __init__(
        self,
        target_sr: int = 22050,
        mono: bool = True,
        segment_duration: float = 5.0,
        silence_top_db: int = 20,
        normalize_loudness: bool = True,
    ):
        self.target_sr = target_sr
        self.mono = mono
        self.segment_duration = segment_duration
        self.target_length = int(target_sr * segment_duration)
        self.silence_top_db = silence_top_db
        self.normalize_loudness = normalize_loudness
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Loads WAV or MP3 audio file, resamples, and converts to mono."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")
        try:
            # Librosa automatically handles WAV and MP3
            y, sr = librosa.load(file_path, sr=self.target_sr, mono=self.mono)
            if y is None or len(y) == 0:
                raise ValueError(f"Loaded empty audio array from {file_path}")
            return y, sr
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise e
    def remove_silence(self, y: np.ndarray) -> np.ndarray:
        """Removes silent portions from audio signal using top dB threshold."""
        if len(y) == 0:
            return y
        non_silent_intervals = librosa.effects.split(y, top_db=self.silence_top_db)
        if len(non_silent_intervals) == 0:
            return y
        y_trimmed = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        return y_trimmed
    def reduce_noise(self, y: np.ndarray) -> np.ndarray:
        """Applies basic spectral noise reduction / smoothing."""
        if len(y) < 512:
            return y
        # Simple moving average spectral noise smoothing filter
        stft = librosa.stft(y)
        magnitude, phase = librosa.magphase(stft)
        noise_floor = np.mean(magnitude, axis=1, keepdims=True) * 0.1
        magnitude_clean = np.maximum(magnitude - noise_floor, 0.0)
        y_clean = librosa.istft(magnitude_clean * phase)
        return y_clean
    def normalize_loudness_signal(self, y: np.ndarray) -> np.ndarray:
        """Normalizes audio signal to peak amplitude 1.0."""
        max_val = np.max(np.abs(y))
        if max_val > 0:
            return y / max_val
        return y
    def pad_or_trim(self, y: np.ndarray) -> np.ndarray:
        """Pads short audio with zeros or trims long audio to exact target sample length."""
        current_len = len(y)
        if current_len == self.target_length:
            return y
        elif current_len < self.target_length:
            pad_width = self.target_length - current_len
            return np.pad(y, (0, pad_width), mode="constant")
        else:
            return y[: self.target_length]
    def segment_audio(
        self, y: np.ndarray, hop_duration: float = 2.5
    ) -> List[np.ndarray]:
        """Splits long audio signal into fixed duration overlapping clips."""
        hop_length = int(self.target_sr * hop_duration)
        segments = []
        if len(y) <= self.target_length:
            segments.append(self.pad_or_trim(y))
        else:
            for start in range(0, len(y) - self.target_length + 1, hop_length):
                seg = y[start : start + self.target_length]
                segments.append(seg)
            # Catch leftover tail if meaningful length
            if len(y) % hop_length != 0 and len(segments) == 0:
                segments.append(self.pad_or_trim(y[-self.target_length :]))
        return segments
    def process_file(
        self, file_path: Union[str, Path], hop_duration: float = 2.5
    ) -> List[np.ndarray]:
        """Full end-to-end preprocessing pipeline for an audio file."""
        y, sr = self.load_audio(file_path)
        y = self.remove_silence(y)
        y = self.reduce_noise(y)
        if self.normalize_loudness:
            y = self.normalize_loudness_signal(y)
        segments = self.segment_audio(y, hop_duration=hop_duration)
        return segments