from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .audio_config import DEFAULT_CONFIG
    from .audio_features import AudioAugmentationConfig, mel_spectrogram_windows
except ImportError:
    from audio_config import DEFAULT_CONFIG
    from audio_features import AudioAugmentationConfig, mel_spectrogram_windows


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def iter_audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)


def collect_dataset(dataset_dir: Path, max_per_category: int | None = None, seed: int = 42) -> list[dict[str, object]]:
    original_dir = dataset_dir / "original"
    splicing_dir = dataset_dir / "Splicing_Labeled"
    if not original_dir.exists():
        raise FileNotFoundError(f"Original audio folder not found: {original_dir}")
    if not splicing_dir.exists():
        raise FileNotFoundError(f"Splicing audio folder not found: {splicing_dir}")

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    if max_per_category is None:
        original_files = iter_audio_files(original_dir)
        splicing_files = iter_audio_files(splicing_dir)
        balance_num = min(len(original_files), len(splicing_files))
    else:
        balance_num = max_per_category

    categories = [
        ("original", 0, original_dir),
        ("splicing", 1, splicing_dir),
    ]
    for category, label, folder in categories:
        files = iter_audio_files(folder)
        if balance_num is not None and len(files) > balance_num:
            selected_indices = rng.choice(len(files), size=balance_num, replace=False)
            files = [files[index] for index in sorted(selected_indices)]
        for audio_path in files:
            rows.append(
                {
                    "audio_path": str(audio_path),
                    "label": label,
                    "label_name": "original" if label == 0 else "edited",
                    "source_category": category,
                }
            )
    if not rows:
        raise RuntimeError(f"No audio files found in {dataset_dir}")
    return rows


def build_mel_dataset(
    dataset_dir: Path,
    output_dir: Path,
    max_per_category: int | None = None,
    seed: int = 42,
    augment_copies: int = 0,
) -> pd.DataFrame:
    augment_copies = max(0, augment_copies)
    output_dir.mkdir(parents=True, exist_ok=True)
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    rows = collect_dataset(dataset_dir, max_per_category=max_per_category, seed=seed)
    augmentation_config = AudioAugmentationConfig() if augment_copies > 0 else None
    window_index = 0
    for index, row in enumerate(tqdm(rows, desc="Creating mel spectrograms")):
        audio_path = Path(str(row["audio_path"]))
        label_name = str(row["source_category"])
        variants: list[tuple[str, AudioAugmentationConfig | None]] = [("none", None)]
        variants.extend(
            (f"waveform_{copy_index:02d}", augmentation_config)
            for copy_index in range(1, augment_copies + 1)
        )

        for augmentation_name, augmentation in variants:
            mel_windows = mel_spectrogram_windows(
                audio_path,
                DEFAULT_CONFIG,
                augmentation_config=augmentation,
            )
            for win_idx, mel in enumerate(mel_windows):
                if augmentation_name == "none":
                    safe_stem = f"{index:06d}_{win_idx:03d}_{audio_path.stem}"
                else:
                    safe_stem = f"{index:06d}_{augmentation_name}_{win_idx:03d}_{audio_path.stem}"
                mel_path = array_dir / label_name / f"{safe_stem}.npy"
                mel_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(mel_path, mel)

                manifest_rows.append(
                    {
                        "audio_path": str(audio_path),
                        "mel_path": str(mel_path),
                        "label": int(row["label"]),
                        "label_name": str(row["label_name"]),
                        "source_category": label_name,
                        "window_index": win_idx,
                        "augmentation": augmentation_name,
                    }
                )
                window_index += 1

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    feature_config = dict(DEFAULT_CONFIG.__dict__)
    feature_config["augment_copies"] = augment_copies
    feature_config["augmentation"] = augmentation_config.__dict__ if augmentation_config is not None else None
    (output_dir / "feature_config.json").write_text(json.dumps(feature_config, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mel-spectrogram dataset from original and splicing audio.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"), help="Folder with original/ and Splicing_Labeled/.")
    parser.add_argument("--output-dir", type=Path, default=Path("mel_dataset"), help="Output folder for .npy mel files and manifest.csv.")
    parser.add_argument("--max-per-category", type=int, default=None, help="Take at most this many files from each source category.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for category sampling.")
    parser.add_argument("--augment", action="store_true", help="Add one waveform-augmented copy before mel conversion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_mel_dataset(
        args.dataset_dir,
        args.output_dir,
        max_per_category=args.max_per_category,
        seed=args.seed,
        augment_copies=1 if args.augment else 0,
    )
    counts = manifest["source_category"].value_counts().to_dict()
    label_counts = manifest["label_name"].value_counts().to_dict()
    print(f"Saved {len(manifest)} mel spectrograms to {args.output_dir}")
    print(f"Source category counts: {counts}")
    print(f"Binary class counts: {label_counts}")


if __name__ == "__main__":
    main()
