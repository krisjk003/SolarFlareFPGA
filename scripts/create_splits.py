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
    * The train/val carve-out is GROUPED BY ACTIVE REGION (never splits a
      single AR's sequences across both sides) and, within that grouping
      constraint, stratified by class label using `StratifiedGroupKFold`
      (`dataset.val_fraction`, `dataset.split_seed`). This is the fix for a
      real leakage bug: SDOBenchmark has multiple sequences per AR, and a
      plain per-sequence stratified split lets the same AR's magnetic
      signature appear in both train and val, letting the model partly
      memorize AR identity instead of learning flare precursors. A
      post-split assertion guarantees zero AR overlap between train/val;
      it hard-fails rather than silently continuing if that's ever violated.
    * Because the *held-out* split(s) (e.g. "test") are copied through
      unchanged from the original dataset layout, this script also checks
      them for AR overlap with the trainable split and warns loudly if
      found -- that's a second, separate leakage risk this script cannot
      silently fix (it would mean redefining what counts as "test"). Pass
      --drop-overlapping-holdout-ars to remove those ARs from the holdout
      split(s) instead of just warning.

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


def _group_overlap(name_a: str, df_a: pd.DataFrame, name_b: str, df_b: pd.DataFrame,
                    logger, group_col: str = "active_region") -> set:
    """Log (never silently ignore) any Active Region shared between two
    splits. Returns the overlapping set so callers can act on it."""
    overlap = set(df_a[group_col]) & set(df_b[group_col])
    if overlap:
        sample = sorted(overlap)[:20]
        logger.warning(
            "Leakage check: %d Active Region(s) appear in BOTH '%s' and '%s'. "
            "First up to 20: %s", len(overlap), name_a, name_b, sample,
        )
    else:
        logger.info("Leakage check: '%s' and '%s' share zero Active Regions. OK.", name_a, name_b)
    return overlap


def _grouped_stratified_split(
    df: pd.DataFrame, val_fraction: float, seed: int, logger,
    group_col: str = "active_region", label_col: str = "label_index",
) -> Any:
    """Split `df` into train/val such that every row for a given
    `group_col` value (Active Region) lands entirely on one side -- this is
    what prevents the AR-leakage bug where the same Active Region's
    sequences show up in both train and val.

    Primary strategy: StratifiedGroupKFold -- groups by AR *and* tries to
    keep the class-label ratio close to equal across train/val, which
    matters here because the dataset is heavily imbalanced (~4:1
    QUIET:FLARE) and an unstratified grouped split could easily hand val a
    noticeably different ratio than train just from AR-size variance.

    Falls back to GroupShuffleSplit (groups by AR only, no label
    stratification) when there are too few distinct ARs or too few classes
    for StratifiedGroupKFold to run.
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    n_groups = df[group_col].nunique()
    n_classes = df[label_col].nunique()
    n_splits = max(2, round(1 / val_fraction))

    if n_groups < n_splits or n_classes < 2:
        logger.warning(
            "Too few Active Regions (%d) or classes (%d) for StratifiedGroupKFold with "
            "n_splits=%d; falling back to GroupShuffleSplit (grouped by AR, but not "
            "label-stratified -- check the resulting class distribution below).",
            n_groups, n_classes, n_splits,
        )
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, df[label_col], groups=df[group_col]))

    train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

    # Hard safety net: this should be structurally impossible given the
    # grouped splitters above, so treat a violation as a bug, not a warning.
    overlap = set(train_df[group_col]) & set(val_df[group_col])
    if overlap:
        raise RuntimeError(
            f"Internal error: grouped split still produced {len(overlap)} overlapping "
            f"Active Region(s) between train and val: {sorted(overlap)[:10]}. This should "
            "be impossible -- please check your scikit-learn version (needs >=1.1)."
        )

    achieved_val_fraction = len(val_df) / len(df) if len(df) else 0.0
    logger.info(
        "Grouped split achieved val_fraction=%.3f (configured %.3f) across %d val ARs.",
        achieved_val_fraction, val_fraction, val_df[group_col].nunique(),
    )
    return train_df, val_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test split files from the preprocessed inventory.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--drop-overlapping-holdout-ars", action="store_true",
        help="If an Active Region appears in both the trainable split and a holdout split "
             "(e.g. test), remove it from the holdout split instead of just warning. "
             "Shrinks the holdout set and diverges from the dataset's original test "
             "definition, so it's opt-in.",
    )
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
    train_df, val_df = _grouped_stratified_split(trainable_df, val_fraction, seed, logger)
    _group_overlap("train", train_df, "val", val_df, logger)

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
        overlap = _group_overlap(name, holdout_df, train_split_name, trainable_df, logger)

        if overlap and args.drop_overlapping_holdout_ars:
            before = len(holdout_df)
            holdout_df = holdout_df[~holdout_df["active_region"].isin(overlap)].copy()
            logger.warning(
                "--drop-overlapping-holdout-ars: removed %d/%d sequences (%d Active Regions) "
                "from '%s' to eliminate overlap with '%s'. This shrinks and redefines the "
                "holdout set relative to the dataset's original split.",
                before - len(holdout_df), before, len(overlap), name, train_split_name,
            )
        elif overlap:
            logger.warning(
                "'%s' still overlaps '%s' by %d Active Region(s) -- results on '%s' are not a "
                "fully independent estimate until this is addressed. Re-run with "
                "--drop-overlapping-holdout-ars to fix automatically, or handle it manually.",
                name, train_split_name, len(overlap), name,
            )

        holdout_df.to_csv(splits_dir / f"{name}.csv", index=False)
        logger.info(
            "Wrote %s.csv (%d sequences). Class distribution: %s",
            name, len(holdout_df), holdout_df["label_name"].value_counts().to_dict(),
        )

    if val_df.empty:
        logger.warning(
            "val.csv is empty (val_fraction=%.3f on %d sequences). Increase dataset.val_fraction "
            "in config.yaml if you need a non-trivial validation set.", val_fraction, len(trainable_df),
        )


if __name__ == "__main__":
    main()