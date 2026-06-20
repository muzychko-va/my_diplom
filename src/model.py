from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from .audio_config import DEFAULT_CONFIG
except ImportError:
    from audio_config import DEFAULT_CONFIG


class AudioForgeryCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.30), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def create_model(device: torch.device | str = "cpu") -> AudioForgeryCNN:
    model = AudioForgeryCNN()
    return model.to(device)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    metrics: dict[str, Any],
    threshold: float = 0.5,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "AudioForgeryCNN",
            "feature_config": DEFAULT_CONFIG.__dict__,
            "threshold": threshold,
            "metrics": metrics,
            "labels": {"0": "original", "1": "edited"},
        },
        output,
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[AudioForgeryCNN, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = create_model(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict_mel(model: nn.Module, mel: np.ndarray, device: torch.device | str = "cpu") -> float:
    tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)
    logit = model(tensor)
    return float(torch.sigmoid(logit).item())
