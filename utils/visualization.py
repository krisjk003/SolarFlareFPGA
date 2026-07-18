"""
utils/visualization.py

Plotting helpers: loss/accuracy curves, confusion matrix, and ROC curve. All
functions save a PNG to disk rather than calling plt.show(), since this
pipeline is designed to run headless (training servers, notebooks, CI, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # Headless backend: never requires a display.
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before this import)
import numpy as np  # noqa: E402


def plot_training_curves(history: Dict[str, List[float]], output_dir: Path) -> None:
    """Save 'Loss vs Epoch' and 'Accuracy vs Epoch' plots from a Trainer's
    history dict (keys: train_loss, val_loss, train_accuracy, val_accuracy).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_vs_epoch.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_epoch.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm: Sequence[Sequence[int]], class_names: List[str], output_path: Path) -> None:
    """Save a heat-map style confusion matrix plot with cell counts annotated."""
    cm_array = np.asarray(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_array, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm_array.max() / 2.0 if cm_array.size else 0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            plt.text(
                j, i, str(cm_array[i, j]), ha="center", va="center",
                color="white" if cm_array[i, j] > threshold else "black",
            )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(fpr: Sequence[float], tpr: Sequence[float], auc: float, output_path: Path) -> None:
    """Save an ROC curve plot (binary classification only)."""
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()