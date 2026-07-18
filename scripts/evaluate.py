"""
scripts/evaluate.py

Loads a trained checkpoint and computes the full metric suite (accuracy,
macro precision/recall/F1, confusion matrix, classification report, and for
binary problems ROC-AUC / ROC curve) on a chosen split file produced by
create_splits.py -- by default `test`, i.e. whatever holdout split was
copied through unchanged from the original dataset.

Metric computation and plotting are never re-implemented here: this script
only orchestrates `utils.metrics.compute_classification_metrics` and
`utils.visualization.plot_confusion_matrix` / `plot_roc_curve`.

Usage:
    python scripts/evaluate.py --config configs/config.yaml
    python scripts/evaluate.py --config configs/config.yaml --split test --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets.scanner import DatasetScanner  # noqa: E402
from datasets.sdo_dataset import SDOBenchmarkDataset  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.image_utils import SUPPORTED_EXTENSIONS  # noqa: E402
from utils.logger import setup_logger  # noqa: E402
from utils.metrics import compute_classification_metrics  # noqa: E402
from utils.visualization import plot_confusion_matrix, plot_roc_curve  # noqa: E402


def _import_build_model():
    try:
        from models import build_model  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import `build_model` from a `models` package at the project root. "
            "evaluate.py needs the same models/ package described in train.py's docstring "
            "(build_model(model_config: dict, num_classes: int) -> torch.nn.Module) to "
            "reconstruct the model architecture before loading checkpoint weights into it."
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
            logger.warning("Sequence %s has no valid images at eval time; skipping.", row.sequence_id)
            continue
        records.append(record)
        label_map[record.sequence_id] = int(row.label_index)
    return SDOBenchmarkDataset(records, label_map, image_size, mean, std, frame_mode)


class Evaluator:
    """Owns model reconstruction + checkpoint loading and the forward pass
    used to collect predictions for metric computation."""

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.model: torch.nn.Module = None  # type: ignore[assignment]
        self.class_names: List[str] = []

    def load_model(self, checkpoint_path: Path) -> None:
        build_model = _import_build_model()
        checkpoint_path = Path(checkpoint_path)
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.class_names = raw_checkpoint.get("class_names")
        if not self.class_names:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain 'class_names'; cannot determine "
                "num_classes to reconstruct the model architecture."
            )
        model_cfg = self.config.get("model", {})
        self.model = build_model(model_cfg, len(self.class_names)).to(self.device)
        load_checkpoint(checkpoint_path, self.model)
        self.model.eval()

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (y_true, y_pred, y_prob) as numpy arrays."""
        all_true, all_pred, all_prob = [], [], []
        for images, labels in loader:
            images = images.to(self.device)
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
            all_true.append(labels.numpy())
            all_pred.append(probs.argmax(dim=1).cpu().numpy())
            all_prob.append(probs.cpu().numpy())
        return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_prob)

    def run(self, split_name: str, splits_dir: Path, scanner: DatasetScanner, loader_kwargs: Dict[str, Any],
            image_size: Tuple[int, int], mean: List[float], std: List[float], frame_mode: str, logger) -> Dict[str, Any]:
        dataset = _load_split_dataset(splits_dir / f"{split_name}.csv", scanner, image_size, mean, std, frame_mode, logger)
        loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
        y_true, y_pred, y_prob = self.predict(loader)
        metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names=self.class_names)
        metrics["num_samples"] = len(dataset)
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on a dataset split.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", type=str, default="test", help="Split file name (without .csv) under paths.splits_dir")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: <checkpoint_dir>/best_model.pth)")
    args = parser.parse_args()

    try:
        config = load_config(Path(args.config))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: could not load config: {exc}", file=sys.stderr)
        sys.exit(1)

    paths_cfg = config.get("paths", {})
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    logger = setup_logger(Path(paths_cfg.get("log_dir", "logs")), "evaluate")
    device = resolve_device(training_cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(paths_cfg.get("checkpoint_dir", "checkpoints")) / "best_model.pth"
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s. Train a model with train.py first, or pass --checkpoint.", checkpoint_path)
        sys.exit(1)

    try:
        evaluator = Evaluator(config, device)
        evaluator.load_model(checkpoint_path)
    except (ImportError, KeyError, FileNotFoundError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    logger.info("Loaded model from %s. Classes: %s", checkpoint_path, evaluator.class_names)

    dataset_root = Path(dataset_cfg["root"])
    extensions = dataset_cfg.get("image_extensions", list(SUPPORTED_EXTENSIONS))
    scanner = DatasetScanner(dataset_root, extensions)
    image_size = tuple(dataset_cfg.get("image_size", [224, 224]))
    mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
    std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
    frame_mode = dataset_cfg.get("frame_mode", "last")
    loader_kwargs = {"batch_size": int(training_cfg.get("batch_size", 32)), "num_workers": int(training_cfg.get("num_workers", 4))}
    splits_dir = Path(paths_cfg.get("splits_dir", "data/splits"))

    try:
        metrics = evaluator.run(args.split, splits_dir, scanner, loader_kwargs, image_size, mean, std, frame_mode, logger)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(
        "Split '%s' (%d samples): accuracy=%.4f precision_macro=%.4f recall_macro=%.4f f1_macro=%.4f",
        args.split, metrics["num_samples"], metrics["accuracy"], metrics["precision_macro"],
        metrics["recall_macro"], metrics["f1_macro"],
    )
    print(metrics["classification_report"])

    report_dir = Path(paths_cfg.get("report_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_confusion_matrix(metrics["confusion_matrix"], evaluator.class_names, report_dir / f"confusion_matrix_{args.split}.png")
    if "roc_curve" in metrics:
        plot_roc_curve(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], metrics["roc_auc"], report_dir / f"roc_curve_{args.split}.png")

    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    serializable["classification_report_text"] = metrics["classification_report"]
    report_path = report_dir / f"evaluate_{args.split}_report_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved metrics report to %s (plots alongside it in %s).", report_path, report_dir)


if __name__ == "__main__":
    main()