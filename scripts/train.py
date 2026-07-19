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

Imbalance handling (FLARE is a small minority of sequences): this script
supports two independent, config-gated mitigations, both on by default --
class-weighted CrossEntropyLoss (`training.class_weighting`) and a
WeightedRandomSampler on the train loader only (`training.weighted_sampler`)
-- plus TSS/FLARE-F1-based best-checkpoint selection (`training.checkpoint_metric`,
default "tss", falling back to FLARE F1 when TSS isn't computable and
finally to validation loss if neither FLARE metric applies at all). See the
Trainer class and the two config-gated sections in main() below for details.

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
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402
from torchvision import transforms  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

from datasets.scanner import DatasetScanner  # noqa: E402
from datasets.sdo_dataset import SDOBenchmarkDataset  # noqa: E402
from utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.image_utils import SUPPORTED_EXTENSIONS  # noqa: E402
from utils.logger import setup_logger  # noqa: E402
from utils.metrics import compute_classification_metrics  # noqa: E402
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
    transform: Optional[Any] = None,
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
    return SDOBenchmarkDataset(records, label_map, image_size, mean, std, frame_mode, transform=transform)


def _best_epoch_index(history: Dict[str, List[float]]) -> Tuple[int, str, float]:
    """Mirrors Trainer's checkpoint-selection priority (tss -> flare_f1 ->
    neg_val_loss) so the end-of-run summary reports the epoch that was
    actually saved as best_model.pth, not just the lowest val_loss epoch.
    """
    for key, negate, name in (
        ("val_tss", False, "tss"),
        ("val_flare_f1", False, "flare_f1"),
        ("val_loss", True, "neg_val_loss"),
    ):
        candidates = [(-v if negate else v, i) for i, v in enumerate(history.get(key, [])) if v is not None]
        if candidates:
            best_value, best_idx = max(candidates)
            return best_idx, name, (-best_value if negate else best_value)
    return 0, "n/a", float("nan")


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
        checkpoint_metric: str = "tss",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scaler = scaler
        # Which metric decides "is_best" -- see _select_checkpoint_metric.
        self.checkpoint_metric = checkpoint_metric

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
    def validate(self, loader: DataLoader, class_names: List[str]) -> Dict[str, Optional[float]]:
        """Runs one validation pass. Returns a dict with `loss` and
        `accuracy` (always present), plus `flare_precision`, `flare_recall`,
        `flare_f1`, and `tss` -- computed via the same
        `utils.metrics.compute_classification_metrics` evaluate.py uses, so
        training-time and evaluation-time numbers agree. Those four keys
        are `None` (not zero) when `class_names` doesn't describe the
        project's binary FLARE/QUIET scheme, so callers can tell "not
        applicable" from "zero score".
        """
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

        metrics: Dict[str, Optional[float]] = {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "flare_precision": None,
            "flare_recall": None,
            "flare_f1": None,
            "tss": None,
        }
        if all_true:
            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            extra = compute_classification_metrics(y_true, y_pred, y_prob=None, class_names=class_names)
            for key in ("flare_precision", "flare_recall", "flare_f1", "tss"):
                if key in extra:
                    metrics[key] = extra[key]
        return metrics

    def _select_checkpoint_metric(self, val_metrics: Dict[str, Optional[float]]) -> Tuple[float, str]:
        """Resolve which validation metric drives `is_best` this epoch, per
        the project's checkpoint-selection policy: TSS by default (or
        FLARE F1 if `self.checkpoint_metric == "flare_f1"`), falling back to
        FLARE F1 when TSS isn't available, and finally to negated validation
        loss if neither FLARE metric applies at all (e.g. not a binary
        FLARE/QUIET run) -- so a "best" checkpoint still gets saved instead
        of never updating. Returns (value_where_higher_is_better, name).
        """
        if self.checkpoint_metric == "flare_f1" and val_metrics.get("flare_f1") is not None:
            return val_metrics["flare_f1"], "flare_f1"
        if val_metrics.get("tss") is not None:
            return val_metrics["tss"], "tss"
        if val_metrics.get("flare_f1") is not None:
            return val_metrics["flare_f1"], "flare_f1"
        return -val_metrics["loss"], "neg_val_loss"

    @staticmethod
    def _history_best(history: Dict[str, List[float]]) -> float:
        """Recovers the running-best checkpoint-selection value from
        `history` (used when resuming via --resume), applying the same
        tss -> flare_f1 -> neg_val_loss priority as
        `_select_checkpoint_metric`, so resumed training doesn't reset (or
        wrongly inflate) the "best so far" baseline.
        """
        for key, negate in (("val_tss", False), ("val_flare_f1", False), ("val_loss", True)):
            values = [v for v in history.get(key, []) if v is not None]
            if values:
                return max(-v for v in values) if negate else max(values)
        return float("-inf")

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
        # Additive keys: present even when resuming from a checkpoint saved
        # before this update (old history dicts won't have them yet).
        history.setdefault("val_tss", [])
        history.setdefault("val_flare_f1", [])

        best_metric_value = self._history_best(history)
        best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")  # kept for reference/state compatibility
        epochs_without_improvement = 0

        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader, class_names)
            val_loss, val_acc = val_metrics["loss"], val_metrics["accuracy"]
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)
            history["val_tss"].append(val_metrics["tss"])
            history["val_flare_f1"].append(val_metrics["flare_f1"])

            metric_value, metric_name = self._select_checkpoint_metric(val_metrics)
            is_best = metric_value > best_metric_value
            if is_best:
                best_metric_value = metric_value
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            best_val_loss = min(best_val_loss, val_loss)  # reference only; no longer drives is_best

            if logger:
                logger.info(
                    "Epoch %d/%d (%.1fs) - train_loss=%.4f train_acc=%.4f - val_loss=%.4f val_acc=%.4f "
                    "val_tss=%s val_flare_f1=%s%s",
                    epoch, num_epochs, elapsed, train_loss, train_acc, val_loss, val_acc,
                    f"{val_metrics['tss']:.4f}" if val_metrics["tss"] is not None else "n/a",
                    f"{val_metrics['flare_f1']:.4f}" if val_metrics["flare_f1"] is not None else "n/a",
                    " (best)" if is_best else "",
                )
                if is_best:
                    logger.info("New best checkpoint (epoch %d): %s=%.4f", epoch, metric_name, metric_value)

            state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
                "history": history,
                "class_names": class_names,
                "config": config,
                "best_val_loss": best_val_loss,
                # Additive, backward-compatible: which metric/value counts as
                # "best" under the new policy. Existing keys above are
                # unchanged, so evaluate.py's checkpoint loading keeps working.
                "checkpoint_metric": metric_name,
                "best_checkpoint_metric_value": best_metric_value,
            }
            save_checkpoint(state, is_best, self.checkpoint_dir)

            if epochs_without_improvement >= early_stopping_patience:
                if logger:
                    logger.info(
                        "Early stopping: no %s improvement for %d epochs.", metric_name, early_stopping_patience
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

    # Training-time augmentations, applied only to the training split. Every
    # transform below operates on the deterministic (H, W, 3) PIL image that
    # `SDOBenchmarkDataset` builds from `preprocess_image()` -- i.e. after
    # loading, 3-channel conversion, and resizing, but before normalisation.
    # Kept mild/conservative since SDOBenchmark patches don't have a strong
    # canonical "up" orientation but are also not natural photographs.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),        
        transforms.RandomRotation(degrees=5),        
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # Validation must reflect real inference-time input, so no augmentation --
    # only the same deterministic ToTensor()/Normalize() training also ends with.
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    try:
        train_dataset = _load_split_dataset(
            splits_dir / "train.csv", scanner, image_size, mean, std, frame_mode, logger, transform=train_transform
        )
        val_dataset = _load_split_dataset(
            splits_dir / "val.csv", scanner, image_size, mean, std, frame_mode, logger, transform=val_transform
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))

    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 4))

    # ------------------------------------------------------------------
    # Shared prep for class weighting (below) and weighted sampling (next
    # section): per-class sample counts from the TRAIN split only.
    # ------------------------------------------------------------------
    train_labels = np.array([label for _, label in train_dataset.samples])
    class_counts = np.bincount(train_labels, minlength=num_classes)
    logger.info("Training samples per class: %s", class_counts.tolist())

    # ------------------------------------------------------------------
    # WeightedRandomSampler (training.weighted_sampler, default: true).
    # ------------------------------------------------------------------
    # Complements class-weighted loss below by oversampling FLARE sequences
    # so each epoch actually shows the model more minority-class examples,
    # rather than relying solely on the loss function to compensate. Only
    # ever applied to the TRAIN loader -- val_loader (and evaluate.py's test
    # loader, untouched) must keep reflecting the real class distribution.
    use_weighted_sampler = bool(training_cfg.get("weighted_sampler", True))
    train_sampler = None
    if use_weighted_sampler:
        inv_class_freq = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
        sample_weights = inv_class_freq[train_labels]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        logger.info("WeightedRandomSampler enabled for the train loader (val/test loaders are unaffected).")
    else:
        logger.info("WeightedRandomSampler disabled (training.weighted_sampler=false); using plain shuffling.")

    # shuffle and sampler are mutually exclusive in DataLoader -- shuffle is
    # only True when there's no sampler.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=train_sampler, shuffle=(train_sampler is None),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(model_cfg, num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    # ------------------------------------------------------------------
    # Class-weighted loss (training.class_weighting, default: true).
    # ------------------------------------------------------------------
    # Solar-flare data is heavily imbalanced (FLARE is a small minority of
    # sequences), so an unweighted CrossEntropyLoss lets the model minimize
    # loss by mostly predicting the majority class. Inverse-frequency class
    # weights (sklearn's `compute_class_weight(class_weight="balanced", ...)`,
    # equivalent to n_samples / (n_classes * class_count)) penalize
    # minority-class mistakes more heavily to counteract that.
    use_class_weighting = bool(training_cfg.get("class_weighting", True))
    if use_class_weighting:
        class_weights_np = compute_class_weight(
            class_weight="balanced", classes=np.arange(num_classes), y=train_labels
        )
        logger.info("Inverse-frequency class weights: %s", class_weights_np.tolist())
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    else:
        logger.info("Class weighting disabled (training.class_weighting=false); using plain CrossEntropyLoss.")
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = bool(training_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed-precision training enabled.")

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    # training.checkpoint_metric (default: "tss") -- see Trainer._select_checkpoint_metric
    # for the full tss -> flare_f1 -> neg_val_loss fallback policy.
    checkpoint_metric_cfg = str(training_cfg.get("checkpoint_metric", "tss")).strip().lower()
    trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir, scaler, checkpoint_metric=checkpoint_metric_cfg)

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

    best_idx, best_metric_name, best_metric_value = _best_epoch_index(history)
    logger.info(
        "Training complete. Best epoch: %d (%s=%.4f, val_loss=%.4f, val_acc=%.4f). "
        "Checkpoints in %s, curves in %s.",
        best_idx + 1, best_metric_name, best_metric_value,
        history["val_loss"][best_idx], history["val_accuracy"][best_idx],
        checkpoint_dir, report_dir,
    )


if __name__ == "__main__":
    main()