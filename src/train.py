from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .model import create_model, save_checkpoint
except ImportError:
    from model import create_model, save_checkpoint


class MelDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        mel = np.load(row["mel_path"]).astype(np.float32)
        x = torch.from_numpy(mel).unsqueeze(0)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    probabilities: list[float] = []
    labels: list[int] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += float(loss.item()) * inputs.size(0)
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(targets.cpu().numpy().astype(int).tolist())
    return total_loss / len(loader.dataset), np.array(probabilities), np.array(labels)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | list[list[int]]]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def plot_training(history: dict[str, list[float]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png")
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path, threshold: float = 0.5) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["original", "edited"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    display.ax_.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    if len(np.unique(y_true)) != 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary classifier for audio edit detection.")
    parser.add_argument("--manifest", type=Path, default=Path("mel_dataset/manifest.csv"), help="Path to manifest.csv from build_mel_dataset.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Where to save model and evaluation artifacts.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.manifest)
    if frame["label"].nunique() != 2:
        raise RuntimeError("Training requires both classes: original and edited.")

    train_frame, val_frame = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=frame["label"],
    )
    train_loader = DataLoader(MelDataset(train_frame), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(MelDataset(val_frame), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)
    labels = train_frame["label"].to_numpy()
    negative_count = max(1, int((labels == 0).sum()))
    positive_count = max(1, int((labels == 1).sum()))
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")
    best_state = None

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, y_prob, y_true = evaluate(model, val_loader, criterion, device)
    metrics = compute_metrics(y_true, y_prob)
    metrics["validation_loss"] = float(val_loss)
    metrics["train_size"] = int(len(train_frame))
    metrics["validation_size"] = int(len(val_frame))

    save_checkpoint(args.output_dir / "audio_forgery_cnn.pt", model, metrics)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_training(history, args.output_dir)
    plot_confusion(y_true, y_prob, args.output_dir)
    plot_roc(y_true, y_prob, args.output_dir)

    print(f"Saved model: {args.output_dir / 'audio_forgery_cnn.pt'}")
    print(f"Saved metrics and plots to: {args.output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
