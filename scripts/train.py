"""
scripts/train.py

Trains the flare-classification model on the splits produced by
create_splits.py.

This script never re-implements image loading/preprocessing, label
encoding, or device/checkpoint logic: it wires together the existing
`datasets.sdo_dataset.SDOBenchmarkDataset`, `datasets.scanner.DatasetScanner`
(reused here to rebuild each split's `SequenceRecord`s so images are
re-validated and chronologically ordered exactly once, in one place),
`utils.device.resolve_device`, and `utils.checkpoint.save_checkpoint` /
`load_checkpoint`.

Model construction is delegated to a `models` package at the project root
that must expose:

    build_model(model_config: dict, num_classes: int) -> torch.nn.Module

where `model_config` is the `model:` section of config.yaml (e.g.
{"name": "resnet18", "pretrained": true}) and `num_classes` is derived
automatically from the resolved dataset labels (see preprocess.py). That
package was not present among the uploaded folders -- add it before running
this script; the import guard below explains exactly what is expected if
it's still missing.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --resume checkpoints/last_model.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets.scanner import DatasetScanner  # noqa: E402
from datasets.sdo_dataset import SDOBenchmarkDataset  # noqa: E402
from utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.image_utils import SUPPORTED_EXTENSIONS  # noqa: E402
from utils.logger import setup_logger  # noqa: E402
from utils.visualization import plot_training_curves  # noqa: E402


def _import_build_model():
    try:
        from models import build_model  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import `build_model` from a `models` package at the project root "
            "(expected alongside datasets/, utils/, configs/, scripts/). train.py requires:\n\n"
            "    models/__init__.py exposing:\n"
            "        def build_model(model_config: dict, num_classes: int) -> torch.nn.Module\n\n"
            "`model_config` is the `model:` section of config.yaml (e.g. {'name': 'resnet18', "
            "'pretrained': True}); `num_classes` is derived automatically from data/splits/classes.json. "
            "Add this package before running train.py."
        ) from exc
    return build_model


# Read these columns as plain strings so numeric-looking folder names (e.g.
# an Active Region "00123") never get silently coerced to an int and lose
# their leading zeros / exact on-disk spelling.
_ID_LIKE_COLUMNS = {
    "sequence_id": str, "active_region": str, "sequence_name": str,
    "sequence_path": str, "last_image_path": str, "label_name": str,
}


def _load_split_dataset(
    split_csv: Path, scanner: DatasetScanner, image_size: Tuple[int, int],
    mean: List[float], std: List[float], frame_mode: str, logger,
) -> SDOBenchmarkDataset:
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_csv}. Run `python scripts/create_splits.py --config <config>` first."
        )
    df = pd.read_csv(split_csv, dtype=_ID_LIKE_COLUMNS)
    records = []
    label_map: Dict[str, int] = {}
    for row in df.itertuples(index=False):
        record, _ = scanner.scan_sequence(str(row.active_region), Path(row.sequence_path))
        if not record.image_paths:
            logger.warning("Sequence %s has no valid images at train time; skipping.", row.sequence_id)
            continue
        records.append(record)
        label_map[record.sequence_id] = int(row.label_index)
    return SDOBenchmarkDataset(records, label_map, image_size, mean, std, frame_mode)


class Trainer:
    """Owns the epoch loop, validation, checkpointing, and early stopping.
    Kept separate from argument parsing / dataset construction in main() so
    it can be reused (e.g. from a notebook) without going through the CLI.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: Path,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scaler = scaler

    def train_one_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int,
        class_names: List[str],
        config: Dict[str, Any],
        start_epoch: int = 1,
        history: Optional[Dict[str, List[float]]] = None,
        logger=None,
    ) -> Dict[str, List[float]]:
        history = history or {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
        best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
        epochs_without_improvement = 0

        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if logger:
                logger.info(
                    "Epoch %d/%d (%.1fs) - train_loss=%.4f train_acc=%.4f - val_loss=%.4f val_acc=%.4f%s",
                    epoch, num_epochs, elapsed, train_loss, train_acc, val_loss, val_acc,
                    " (best)" if is_best else "",
                )

            state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
                "history": history,
                "class_names": class_names,
                "config": config,
                "best_val_loss": best_val_loss,
            }
            save_checkpoint(state, is_best, self.checkpoint_dir)

            if epochs_without_improvement >= early_stopping_patience:
                if logger:
                    logger.info(
                        "Early stopping: no val_loss improvement for %d epochs.", early_stopping_patience
                    )
                break

        return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the flare classification model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from.")
    args = parser.parse_args()

    try:
        config = load_config(Path(args.config))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: could not load config: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        build_model = _import_build_model()
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    paths_cfg = config.get("paths", {})
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    logger = setup_logger(Path(paths_cfg.get("log_dir", "logs")), "train")
    device = resolve_device(training_cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    splits_dir = Path(paths_cfg.get("splits_dir", "data/splits"))
    classes_path = splits_dir / "classes.json"
    if not classes_path.exists():
        logger.error("%s not found. Run preprocess.py then create_splits.py first.", classes_path)
        sys.exit(1)
    with classes_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)["classes"]

    num_classes = model_cfg.get("num_classes") or len(class_names)
    if model_cfg.get("num_classes") and model_cfg["num_classes"] != len(class_names):
        logger.warning(
            "model.num_classes=%s in config.yaml does not match %d classes resolved from data (%s); "
            "using the resolved value.", model_cfg.get("num_classes"), len(class_names), class_names,
        )
        num_classes = len(class_names)
    logger.info("Classes (%d): %s", num_classes, class_names)

    dataset_root = Path(dataset_cfg["root"])
    extensions = dataset_cfg.get("image_extensions", list(SUPPORTED_EXTENSIONS))
    scanner = DatasetScanner(dataset_root, extensions)
    image_size = tuple(dataset_cfg.get("image_size", [224, 224]))
    mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
    std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
    frame_mode = dataset_cfg.get("frame_mode", "last")

    try:
        train_dataset = _load_split_dataset(splits_dir / "train.csv", scanner, image_size, mean, std, frame_mode, logger)
        val_dataset = _load_split_dataset(splits_dir / "val.csv", scanner, image_size, mean, std, frame_mode, logger)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))

    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 4))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(model_cfg, num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    class_weights = torch.tensor([17.75, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = bool(training_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed-precision training enabled.")

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir, scaler)

    start_epoch, history = 1, None
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), model, optimizer, scaler)
        start_epoch = checkpoint.get("epoch", 0) + 1
        history = checkpoint.get("history")
        logger.info("Resumed from %s at epoch %d.", args.resume, start_epoch)

    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=int(training_cfg.get("num_epochs", 30)),
        early_stopping_patience=int(training_cfg.get("early_stopping_patience", 7)),
        class_names=class_names, config=config,
        start_epoch=start_epoch, history=history, logger=logger,
    )

    report_dir = Path(paths_cfg.get("report_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, report_dir)

    best_epoch = int(min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])) + 1
    logger.info(
        "Training complete. Best epoch: %d (val_loss=%.4f, val_acc=%.4f). "
        "Checkpoints in %s, curves in %s.",
        best_epoch, history["val_loss"][best_epoch - 1], history["val_accuracy"][best_epoch - 1],
        checkpoint_dir, report_dir,
    )


if __name__ == "__main__":
    main()