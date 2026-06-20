from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    window_duration: float = 4.0
    hop_duration: float = 2.0
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: int = 20
    fmax: int = 8_000

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_duration)

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_duration)

    @property
    def frames(self) -> int:
        return 1 + self.window_samples // self.hop_length


DEFAULT_CONFIG = AudioConfig()
