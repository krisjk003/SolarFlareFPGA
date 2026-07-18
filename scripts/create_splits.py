"""
scripts/create_splits.py

Builds the final train / val / (holdout) split files consumed by train.py
and evaluate.py, from the canonical inventory that preprocess.py already
scanned and label-resolved (this script does not touch the raw images or
re-run label resolution -- it only partitions already-resolved sequences,
so run preprocess.py first).

Split strategy (fully config-driven, nothing hardcoded):
    * `dataset.train_split_name` (default "training") names the split that
      gets carved into train/val. If a split with that name isn't among the
      ones preprocess.py actually found, the largest discovered split is
      used instead and a warning is logged.
    * Every other discovered split (e.g. "test") is copied through as-is --
      it already represents a held-out evaluation set from the original
      dataset and is never mixed into train/val.
    * The train/val carve-out is stratified by class label
      (`dataset.val_fraction`, `dataset.split_seed`), falling back to a
      non-stratified split with a warning if any class has too few samples
      to stratify.

Output:
    <paths.splits_dir>/classes.json
    <paths.splits_dir>/train.csv
    <paths.splits_dir>/val.csv
    <paths.splits_dir>/<other split name>.csv   (one per remaining split)

Usage:
    python scripts/create_splits.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import load_config  # noqa: E402
from utils.logger import setup_logger  # noqa: E402


# Read these columns as plain strings so numeric-looking folder names (e.g.
# an Active Region "00123") never get silently coerced to an int and lose
# their leading zeros / exact on-disk spelling.
_ID_LIKE_COLUMNS = {
    "sequence_id": str, "active_region": str, "sequence_name": str,
    "sequence_path": str, "last_image_path": str, "label_name": str,
}


def _load_inventory(interim_dir: Path, split_name: str) -> pd.DataFrame:
    csv_path = interim_dir / f"{split_name}_inventory.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No inventory found for split '{split_name}' at {csv_path}. "
            "Run `python scripts/preprocess.py --config <config>` first."
        )
    df = pd.read_csv(csv_path, dtype=_ID_LIKE_COLUMNS)
    return df[df["label_status"] == "resolved"].copy()


def _discover_preprocessed_splits(interim_dir: Path) -> List[str]:
    if not interim_dir.exists():
        raise FileNotFoundError(
            f"{interim_dir} does not exist. Run `python scripts/preprocess.py --config <config>` first."
        )
    return sorted(p.stem[: -len("_inventory")] for p in interim_dir.glob("*_inventory.csv"))


def _choose_train_split(splits: List[str], inventories: Dict[str, pd.DataFrame], configured_name: str, logger) -> str:
    for name in splits:
        if name.lower() == configured_name.lower():
            return name
    fallback = max(splits, key=lambda s: len(inventories[s]))
    logger.warning(
        "No split named '%s' (dataset.train_split_name) was found among %s; "
        "using the largest discovered split '%s' as the trainable set instead.",
        configured_name, splits, fallback,
    )
    return fallback


def _stratified_split(df: pd.DataFrame, val_fraction: float, seed: int, logger) -> Any:
    from sklearn.model_selection import train_test_split

    label_counts = df["label_index"].value_counts()
    can_stratify = label_counts.min() >= 2 and len(df) * val_fraction >= len(label_counts)
    try:
        if not can_stratify:
            raise ValueError("insufficient per-class samples to stratify")
        train_df, val_df = train_test_split(
            df, test_size=val_fraction, random_state=seed, stratify=df["label_index"],
        )
    except ValueError as exc:
        logger.warning(
            "Falling back to a non-stratified train/val split (%s). Class balance across "
            "train/val may be uneven -- consider collecting more samples for rare classes.", exc,
        )
        train_df, val_df = train_test_split(df, test_size=val_fraction, random_state=seed)
    return train_df, val_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test split files from the preprocessed inventory.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(Path(args.config))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: could not load config: {exc}", file=sys.stderr)
        sys.exit(1)

    paths_cfg = config.get("paths", {})
    dataset_cfg = config.get("dataset", {})
    logger = setup_logger(Path(paths_cfg.get("log_dir", "logs")), "create_splits")

    interim_dir = Path(paths_cfg.get("interim_dir", "data/interim"))
    splits_dir = Path(paths_cfg.get("splits_dir", "data/splits"))

    try:
        splits = _discover_preprocessed_splits(interim_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    if not splits:
        logger.error("No preprocessed inventory CSVs found in %s.", interim_dir)
        sys.exit(1)

    inventories = {s: _load_inventory(interim_dir, s) for s in splits}
    for name, df in inventories.items():
        if df.empty:
            logger.warning("Split '%s' has zero labeled sequences after preprocessing.", name)

    train_split_name = _choose_train_split(
        splits, inventories, dataset_cfg.get("train_split_name", "training"), logger,
    )
    trainable_df = inventories[train_split_name]
    if trainable_df.empty:
        logger.error("Trainable split '%s' has zero labeled sequences; cannot create train/val splits.", train_split_name)
        sys.exit(1)

    val_fraction = float(dataset_cfg.get("val_fraction", 0.15))
    seed = int(dataset_cfg.get("split_seed", 42))
    train_df, val_df = _stratified_split(trainable_df, val_fraction, seed, logger)

    splits_dir.mkdir(parents=True, exist_ok=True)
    classes_path = interim_dir / "classes.json"
    if classes_path.exists():
        shutil.copyfile(classes_path, splits_dir / "classes.json")

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    logger.info(
        "Wrote train.csv (%d sequences, source split '%s') and val.csv (%d sequences).",
        len(train_df), train_split_name, len(val_df),
    )
    logger.info("Train class distribution: %s", train_df["label_name"].value_counts().to_dict())
    logger.info("Val class distribution: %s", val_df["label_name"].value_counts().to_dict())

    for name in splits:
        if name == train_split_name:
            continue
        holdout_df = inventories[name]
        holdout_df.to_csv(splits_dir / f"{name}.csv", index=False)
        logger.info(
            "Wrote %s.csv (%d sequences, copied through unchanged as a held-out evaluation set). "
            "Class distribution: %s",
            name, len(holdout_df), holdout_df["label_name"].value_counts().to_dict(),
        )

    if val_df.empty:
        logger.warning(
            "val.csv is empty (val_fraction=%.3f on %d sequences). Increase dataset.val_fraction "
            "in config.yaml if you need a non-trivial validation set.", val_fraction, len(trainable_df),
        )


if __name__ == "__main__":
    main()