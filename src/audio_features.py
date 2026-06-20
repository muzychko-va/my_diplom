from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio.transforms as T
import torch_audiomentations

try:
    from .audio_config import AudioConfig, DEFAULT_CONFIG
except ImportError:
    from audio_config import AudioConfig, DEFAULT_CONFIG


@dataclass(frozen=True)
class AudioAugmentationConfig:
    enabled: bool = True
    min_snr_db: float = 20.0
    max_snr_db: float = 35.0
    min_gain_db: float = -6.0
    max_gain_db: float = 6.0
    shift_probability: float = 0.5
    min_shift: float = -0.5
    max_shift: float = 0.5


def peak_normalize(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


class MelSpectrogramExtractor:
    def __init__(
        self,
        config: AudioConfig = DEFAULT_CONFIG,
        device: str | None = None,
        augmentation_config: AudioAugmentationConfig | None = None,
    ):
        self.config = config
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.augmenter = None
        if augmentation_config is not None and augmentation_config.enabled:
            transforms = [
                torch_audiomentations.AddColoredNoise(
                    min_snr_in_db=augmentation_config.min_snr_db,
                    max_snr_in_db=augmentation_config.max_snr_db,
                    p=0.5,
                ),
                torch_audiomentations.Gain(
                    min_gain_in_db=augmentation_config.min_gain_db,
                    max_gain_in_db=augmentation_config.max_gain_db,
                    p=0.5,
                ),
                torch_audiomentations.Shift(
                    min_shift=augmentation_config.min_shift,
                    max_shift=augmentation_config.max_shift,
                    p=augmentation_config.shift_probability,
                ),
            ]
            
            self.augmenter = torch_audiomentations.Compose(transforms=transforms, output_type="tensor")
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.fmin,
            f_max=config.fmax,
            power=2.0,
        ).to(self.device)
        
        self.db_transform = T.AmplitudeToDB(top_db=80.0).to(self.device)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if y.dtype != np.float32:
            y = y.astype(np.float32)
            
        audio_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(self.device)
        
        if self.augmenter is not None:
            audio_tensor = self.augmenter(audio_tensor, sample_rate=self.config.sample_rate)
        
        with torch.no_grad():
            mel = self.mel_transform(audio_tensor)
            mel_db = self.db_transform(mel)
            
            top_db = self.db_transform.top_db
            mel_db = (mel_db + top_db) / top_db
            mel_db = torch.clamp(mel_db, 0.0, 1.0)
            
            current_frames = mel_db.shape[2]
            if current_frames < self.config.frames:
                mel_db = torch.nn.functional.pad(mel_db, (0, self.config.frames - current_frames))
            else:
                mel_db = mel_db[:, :, :self.config.frames]
                
        return mel_db.squeeze(0).squeeze(0).cpu().numpy()


def load_audio(
    path: str | Path,
    config: AudioConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    y, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    y = peak_normalize(y)
    return y.astype(np.float32)


def extract_windows(y: np.ndarray, config: AudioConfig = DEFAULT_CONFIG) -> list[np.ndarray]:
    window_samples = config.window_samples
    hop_samples = config.hop_samples
    windows = []
    start = 0
    while start + window_samples <= len(y):
        windows.append(y[start:start + window_samples])
        start += hop_samples
    if not windows:
        padded = np.zeros(window_samples, dtype=np.float32)
        length = min(len(y), window_samples)
        padded[:length] = y[:length]
        windows.append(padded)
    return windows


_mel_extractor_cache = {}


def mel_spectrogram_from_audio(
    y: np.ndarray,
    config: AudioConfig = DEFAULT_CONFIG,
    augmentation_config: AudioAugmentationConfig | None = None,
) -> np.ndarray:
    cache_key = (config.sample_rate, config.n_fft, config.hop_length, config.n_mels, config.fmin, config.fmax, config.frames)
    if cache_key not in _mel_extractor_cache:
        _mel_extractor_cache[cache_key] = MelSpectrogramExtractor(config, augmentation_config=augmentation_config)
    return _mel_extractor_cache[cache_key](y)


def mel_spectrogram(
    path: str | Path,
    config: AudioConfig = DEFAULT_CONFIG,
    augmentation_config: AudioAugmentationConfig | None = None,
) -> np.ndarray:
    y = load_audio(path, config)
    return mel_spectrogram_from_audio(y, config, augmentation_config=augmentation_config)


def mel_spectrogram_windows(
    path: str | Path,
    config: AudioConfig = DEFAULT_CONFIG,
    augmentation_config: AudioAugmentationConfig | None = None,
) -> list[np.ndarray]:
    y = load_audio(path, config)
    windows = extract_windows(y, config)
    return [mel_spectrogram_from_audio(w, config, augmentation_config=augmentation_config) for w in windows]


def save_mel_image(mel: np.ndarray, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 4), dpi=120)
    ax = fig.add_subplot(111)
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

