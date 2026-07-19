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

Optional post-hoc threshold optimization (binary FLARE/QUIET problems
only): pass --sweep-thresholds to additionally sweep the decision threshold
applied to the FLARE-class probability from 0.05 to 0.95 in steps of 0.05,
reporting accuracy/precision/recall/F1/TSS at each threshold, the
best-F1 and best-TSS thresholds, and a saved CSV. This never changes the
default (no-flag) evaluation output -- it's a strictly additive step that
runs after the existing metrics/plots/report are already computed and
written exactly as before.

Separately, --threshold FLOAT (also binary FLARE/QUIET problems only)
replaces the standard argmax decision rule with an explicit threshold on
the FLARE-class probability when computing the predictions that feed
accuracy/precision/recall/F1/confusion matrix/classification report.
Unlike --sweep-thresholds, this DOES change the evaluation output -- that
is the point, once you already know which threshold you want to use.
ROC-AUC / ROC curve are unaffected either way, since they only depend on
predicted probabilities, not on the thresholded predictions.

Intended workflow: pick a threshold on a validation split with
`--split val --sweep-thresholds`, then evaluate exactly once on the test
split with `--split test --threshold <best_threshold>`. Don't optimize the
threshold on the test set.

Usage:
    python scripts/evaluate.py --config configs/config.yaml
    python scripts/evaluate.py --config configs/config.yaml --split test --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --config configs/config.yaml --split val --sweep-thresholds
    python scripts/evaluate.py --config configs/config.yaml --split test --threshold 0.35
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# 0.05, 0.10, ..., 0.95 -- built with round() to avoid float drift from repeated addition.
_THRESHOLD_SWEEP_VALUES: List[float] = [round(0.05 * i, 2) for i in range(1, 20)]

_THRESHOLD_SWEEP_FIELDS = ["threshold", "accuracy", "precision", "recall", "f1", "fpr", "tss", "tp", "fp", "tn", "fn"]


def _find_flare_index(class_names: List[str]) -> int:
    """Locate the FLARE class by name rather than assuming a column index.

    This matters because class indices come from `sorted(set(class_names))`
    upstream (see preprocess.py), so for the standard binary scheme
    ["FLARE", "QUIET"] the FLARE probability lives at column 0, NOT column 1
    -- the opposite of the "column 1 = positive class" convention
    utils/metrics.py uses for ROC-AUC (which is fine there, since binary
    ROC-AUC is symmetric to that choice; it is NOT fine for a threshold
    sweep, where precision/recall/F1 absolutely depend on using the right
    column).
    """
    for idx, name in enumerate(class_names):
        if name.strip().upper() == "FLARE":
            return idx
    raise ValueError(
        f"--sweep-thresholds requires a class literally named 'FLARE' among the resolved "
        f"classes; found {class_names}. This feature assumes the project's binary "
        "flare_class_binning scheme (FLARE vs QUIET) and does not apply to multiclass runs."
    )


def _require_binary_flare_index(class_names: List[str]) -> int:
    """Validate that `class_names` describes a binary FLARE/QUIET scheme and
    return the index of the FLARE class.

    Shared by --threshold and --sweep-thresholds, both of which only make
    sense for this binary scheme (see module docstring). `_find_flare_index`
    on its own only checks that a 'FLARE' class exists; this also checks
    there are exactly two classes, since both features interpret the other
    class as QUIET.
    """
    if len(class_names) != 2:
        raise ValueError(
            "--threshold/--sweep-thresholds require a binary FLARE/QUIET classification "
            f"(2 classes); found {len(class_names)} classes: {class_names}."
        )
    return _find_flare_index(class_names)


def _apply_threshold_predictions(
    y_prob: np.ndarray, flare_index: int, other_index: int, threshold: float
) -> np.ndarray:
    """Recompute predicted class labels from the FLARE-class probability
    column using an explicit threshold, instead of the standard argmax rule
    used by `Evaluator.predict`. Binary FLARE/QUIET problems only -- see
    `_require_binary_flare_index`."""
    flare_scores = y_prob[:, flare_index]
    return np.where(flare_scores >= threshold, flare_index, other_index).astype(int)


def _threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, flare_index: int) -> List[Dict[str, float]]:
    """Recompute accuracy/precision/recall/F1/TSS at each candidate
    threshold applied to the FLARE-class probability column, independent of
    whatever threshold (implicitly 0.5, via argmax) produced `y_pred` in the
    default evaluation above."""
    y_true_is_flare = (y_true == flare_index).astype(int)
    flare_scores = y_prob[:, flare_index]

    rows: List[Dict[str, float]] = []
    for threshold in _THRESHOLD_SWEEP_VALUES:
        y_pred_is_flare = (flare_scores >= threshold).astype(int)

        tp = int(((y_pred_is_flare == 1) & (y_true_is_flare == 1)).sum())
        fp = int(((y_pred_is_flare == 1) & (y_true_is_flare == 0)).sum())
        tn = int(((y_pred_is_flare == 0) & (y_true_is_flare == 0)).sum())
        fn = int(((y_pred_is_flare == 0) & (y_true_is_flare == 1)).sum())

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # a.k.a. TPR
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tss = recall - fpr  # True Skill Statistic, standard in solar flare forecasting literature

        rows.append({
            "threshold": threshold, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "fpr": fpr, "tss": tss,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return rows


def _print_threshold_table(rows: List[Dict[str, float]]) -> None:
    header = f"{'thresh':>7} {'acc':>7} {'prec':>7} {'recall':>7} {'f1':>7} {'fpr':>7} {'tss':>7}   {'tp':>4} {'fp':>4} {'tn':>4} {'fn':>4}"
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: r["threshold"]):
        print(
            f"{row['threshold']:>7.2f} {row['accuracy']:>7.4f} {row['precision']:>7.4f} "
            f"{row['recall']:>7.4f} {row['f1']:>7.4f} {row['fpr']:>7.4f} {row['tss']:>7.4f}   "
            f"{row['tp']:>4d} {row['fp']:>4d} {row['tn']:>4d} {row['fn']:>4d}"
        )


def _save_threshold_sweep_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_THRESHOLD_SWEEP_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["threshold"]):
            writer.writerow(row)


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

    def run(
        self, split_name: str, splits_dir: Path, scanner: DatasetScanner, loader_kwargs: Dict[str, Any],
        image_size: Tuple[int, int], mean: List[float], std: List[float], frame_mode: str, logger,
        threshold: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """Runs inference on `split_name` and computes the standard metric
        suite. Returns (metrics, y_true, y_pred, y_prob); the raw arrays are
        exposed so callers (e.g. --sweep-thresholds) can do further
        threshold-based analysis without re-running inference.

        If `threshold` is given, predictions are recomputed from the
        FLARE-class probability using that threshold instead of the default
        argmax rule (binary FLARE/QUIET problems only); this changes the
        returned `y_pred` and every metric derived from it (accuracy,
        precision/recall/F1, confusion matrix, classification report). ROC
        metrics are unaffected, since they only depend on `y_prob`.
        """
        dataset = _load_split_dataset(splits_dir / f"{split_name}.csv", scanner, image_size, mean, std, frame_mode, logger)
        loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
        y_true, y_pred, y_prob = self.predict(loader)

        if threshold is not None:
            flare_index = _require_binary_flare_index(self.class_names)
            other_index = 1 - flare_index
            y_pred = _apply_threshold_predictions(y_prob, flare_index, other_index, threshold)

        metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names=self.class_names)
        metrics["num_samples"] = len(dataset)
        if threshold is not None:
            metrics["applied_threshold"] = threshold
        return metrics, y_true, y_pred, y_prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on a dataset split.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", type=str, default="test", help="Split file name (without .csv) under paths.splits_dir")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: <checkpoint_dir>/best_model.pth)")
    parser.add_argument(
        "--sweep-thresholds", action="store_true",
        help="After the normal evaluation, sweep the decision threshold on the FLARE-class "
             "probability (binary FLARE/QUIET problems only) from 0.05 to 0.95 in steps of "
             "0.05, print the resulting table, save reports/threshold_sweep_<split>.csv, and "
             "report the best-F1 and best-TSS thresholds. Additive: never changes the default "
             "metrics/plots/report.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold applied to the FLARE-class probability when converting "
             "probabilities into predictions (binary FLARE/QUIET problems only), replacing "
             "the default argmax rule (~0.5). Does not affect ROC-AUC/ROC curve. Typically "
             "set to the best threshold found via --sweep-thresholds on a validation split, "
             "then used once on the test split.",
    )
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
        metrics, y_true, y_pred, y_prob = evaluator.run(
            args.split, splits_dir, scanner, loader_kwargs, image_size, mean, std, frame_mode, logger,
            threshold=args.threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(
        "Split '%s' (%d samples): accuracy=%.4f precision_macro=%.4f recall_macro=%.4f f1_macro=%.4f",
        args.split, metrics["num_samples"], metrics["accuracy"], metrics["precision_macro"],
        metrics["recall_macro"], metrics["f1_macro"],
    )
    if args.threshold is not None:
      logger.info(
        "Evaluating with FLARE decision threshold = %.2f",
        args.threshold,
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

    if args.sweep_thresholds:
        if args.split.lower() == "test":
          logger.error(
            "Threshold optimization must be performed on the validation split, "
            "not on the test split. Run with '--split val --sweep-thresholds' "
            "to choose a threshold, then evaluate the test set once using "
            "'--threshold <best_threshold>'."
            )
          sys.exit(1)
        try:
            flare_index = _require_binary_flare_index(evaluator.class_names)
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(1)

        sweep_rows = _threshold_sweep(y_true, y_prob, flare_index)
        _print_threshold_table(sweep_rows)

        sweep_csv_path = report_dir / f"threshold_sweep_{args.split}.csv"
        _save_threshold_sweep_csv(sweep_rows, sweep_csv_path)

        best_f1_row = max(sweep_rows, key=lambda r: r["f1"])
        best_tss_row = max(sweep_rows, key=lambda r: r["tss"])
        logger.info(
            "Best-F1 threshold: %.2f (f1=%.4f, precision=%.4f, recall=%.4f, accuracy=%.4f)",
            best_f1_row["threshold"], best_f1_row["f1"], best_f1_row["precision"],
            best_f1_row["recall"], best_f1_row["accuracy"],
        )
        logger.info(
            "Best-TSS threshold: %.2f (tss=%.4f, recall=%.4f, fpr=%.4f, accuracy=%.4f)",
            best_tss_row["threshold"], best_tss_row["tss"], best_tss_row["recall"],
            best_tss_row["fpr"], best_tss_row["accuracy"],
        )
        logger.info("Saved threshold sweep to %s", sweep_csv_path)


if __name__ == "__main__":
    main()