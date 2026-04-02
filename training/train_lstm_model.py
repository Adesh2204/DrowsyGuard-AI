import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataset_loader import EyeSequenceDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.lstm_model import DROWSINESS_LABELS, DrowsinessLSTM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the temporal Bidirectional LSTM drowsiness classifier.")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data" / "sequences")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "training" / "outputs" / "lstm_model")
    parser.add_argument("--weights-output", type=Path, default=ROOT_DIR / "backend" / "weights" / "drowsiness_lstm.pth")
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--step-size", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_samples(
    samples: Sequence[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[Tuple[Path, int]], list[Tuple[Path, int]]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    val_count = max(1, int(len(indices) * val_ratio))
    val_index_set = set(indices[:val_count].tolist())

    train_samples = [sample for i, sample in enumerate(samples) if i not in val_index_set]
    val_samples = [sample for i, sample in enumerate(samples) if i in val_index_set]

    return train_samples, val_samples


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(sequences)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_accuracy = total_correct / max(total_samples, 1)
    return epoch_loss, epoch_accuracy, all_targets, all_predictions


def save_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=DROWSINESS_LABELS,
        yticklabels=DROWSINESS_LABELS,
    )
    plt.title("LSTM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.weights_output.parent.mkdir(parents=True, exist_ok=True)

    full_dataset = EyeSequenceDataset(root_dir=args.data_dir, sequence_length=args.sequence_length)
    train_samples, val_samples = split_samples(full_dataset.samples, val_ratio=args.val_ratio, seed=args.seed)

    train_dataset = EyeSequenceDataset(
        root_dir=args.data_dir,
        sequence_length=args.sequence_length,
        samples=train_samples,
    )
    val_dataset = EyeSequenceDataset(
        root_dir=args.data_dir,
        sequence_length=args.sequence_length,
        samples=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrowsinessLSTM(
        input_size=3,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_targets: list[int] = []
    best_predictions: list[int] = []

    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_acc, targets, predictions = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_targets = targets
            best_predictions = predictions
            checkpoint = {
                "state_dict": model.state_dict(),
                "labels": DROWSINESS_LABELS,
                "epoch": epoch + 1,
                "val_accuracy": best_val_acc,
            }
            torch.save(checkpoint, args.weights_output)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    if not best_targets:
        raise RuntimeError("Validation set produced no targets. Check the sequence dataset.")

    accuracy = accuracy_score(best_targets, best_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        best_targets,
        best_predictions,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(best_targets, best_predictions, labels=list(range(len(DROWSINESS_LABELS))))

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "best_val_accuracy": float(best_val_acc),
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    save_curves(history, args.output_dir / "training_curves.png")
    save_confusion_matrix(cm, args.output_dir / "confusion_matrix.png")

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
