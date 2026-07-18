"""
scripts/preprocess.py

Turns the raw, on-disk SDOBenchmark dataset into a small set of canonical
artifacts that every later stage (create_splits.py, train.py, evaluate.py,
predict.py) reads instead of re-scanning the filesystem and re-resolving
labels every time:

    data/interim/classes.json           canonical class list (index order)
    data/interim/manifest.json          per-split scan + label summary
    data/interim/<split>_inventory.csv  one row per sequence, with its
                                         resolved label (if any)

This is also the one place the project-required rule is enforced end to
end: "automatically detect labels ... if they are not found, clearly report
where they are expected instead of assuming them." Label discovery itself
lives entirely in `datasets.label_resolver.LabelResolver` (not
re-implemented here); this script's job is to run it once per split, and to
reconcile class names across splits so a class index means the same thing
everywhere downstream.

Usage:
    python scripts/preprocess.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.label_resolver import LabelResolutionError, LabelResolver, LabelResolutionResult  # noqa: E402
from datasets.scanner import DatasetScanner, SplitInventory  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.image_utils import SUPPORTED_EXTENSIONS  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

INVENTORY_COLUMNS = [
    "sequence_id", "active_region", "sequence_name", "sequence_path",
    "num_images", "last_image_path", "label_index", "label_name", "label_status",
]


def _get_image_extensions(config: Dict[str, Any]) -> List[str]:
    return config.get("dataset", {}).get("image_extensions", list(SUPPORTED_EXTENSIONS))


def _discover_splits(dataset_root: Path) -> List[str]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset.root '{dataset_root}' does not exist.")
    return sorted(p.name for p in dataset_root.iterdir() if p.is_dir())


def _write_inventory_csv(
    path: Path,
    inventory: SplitInventory,
    label_result: Optional[LabelResolutionResult],
    canonical_classes: List[str],
) -> Dict[str, int]:
    """Write one row per sequence, remapping this split's local label
    indices into the canonical class index space. Returns a
    {class_name: count} distribution for the manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    distribution: Dict[str, int] = {name: 0 for name in canonical_classes}

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        for seq in inventory.sequences:
            local_idx = label_result.label_map.get(seq.sequence_id) if label_result else None
            if local_idx is not None:
                class_name = label_result.class_names[local_idx]
                canonical_idx = canonical_classes.index(class_name)
                label_status = "resolved"
                distribution[class_name] += 1
            else:
                canonical_idx = ""
                class_name = ""
                label_status = "unresolved"

            writer.writerow({
                "sequence_id": seq.sequence_id,
                "active_region": seq.active_region,
                "sequence_name": seq.sequence_name,
                "sequence_path": str(seq.sequence_path),
                "num_images": len(seq.image_paths),
                "last_image_path": str(seq.image_paths[-1]) if seq.image_paths else "",
                "label_index": canonical_idx,
                "label_name": class_name,
                "label_status": label_status,
            })
    return distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan the raw dataset and resolve labels into canonical artifacts.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(Path(args.config))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: could not load config: {exc}", file=sys.stderr)
        sys.exit(1)

    paths_cfg = config.get("paths", {})
    logger = setup_logger(Path(paths_cfg.get("log_dir", "logs")), "preprocess")

    dataset_root = Path(config["dataset"]["root"])
    extensions = _get_image_extensions(config)

    try:
        splits = _discover_splits(dataset_root)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    if not splits:
        logger.error("No split subdirectories found under %s.", dataset_root)
        sys.exit(1)
    logger.info("Discovered %d split(s) under %s: %s", len(splits), dataset_root, splits)

    scanner = DatasetScanner(dataset_root, extensions)
    resolver = LabelResolver(config)

    inventories: Dict[str, SplitInventory] = {}
    label_results: Dict[str, Optional[LabelResolutionResult]] = {}
    label_errors: Dict[str, str] = {}

    for split_name in splits:
        inventory = scanner.scan_split(split_name)
        inventories[split_name] = inventory
        if not inventory.sequences:
            logger.warning("Split '%s' has zero usable sequences after scanning; skipping label resolution.", split_name)
            label_results[split_name] = None
            continue
        try:
            label_results[split_name] = resolver.resolve(inventory)
        except LabelResolutionError as exc:
            label_results[split_name] = None
            label_errors[split_name] = str(exc)
            logger.error("Split '%s': %s", split_name, exc)

    # Reconcile a single canonical class list across every split that
    # resolved successfully, so label_index means the same class everywhere.
    canonical_classes: List[str] = sorted({
        name for result in label_results.values() if result is not None for name in result.class_names
    })

    if not canonical_classes:
        logger.error(
            "No split could be automatically labeled. Preprocessing cannot continue without at "
            "least one labeled split. See the per-split errors above for exactly what was tried "
            "and what is expected (CSV metadata, per-sequence metadata files, folder-name tokens, "
            "or filename tokens) -- or set `dataset.label_source` explicitly in configs/config.yaml."
        )
        sys.exit(1)
    logger.info("Canonical class list (%d classes): %s", len(canonical_classes), canonical_classes)

    interim_dir = Path(paths_cfg.get("interim_dir", "data/interim"))
    interim_dir.mkdir(parents=True, exist_ok=True)

    with (interim_dir / "classes.json").open("w", encoding="utf-8") as f:
        json.dump({"classes": canonical_classes}, f, indent=2)

    manifest: Dict[str, Any] = {"dataset_root": str(dataset_root), "classes": canonical_classes, "splits": {}}

    for split_name in splits:
        inventory = inventories[split_name]
        label_result = label_results[split_name]
        csv_path = interim_dir / f"{split_name}_inventory.csv"

        if inventory.sequences:
            distribution = _write_inventory_csv(csv_path, inventory, label_result, canonical_classes)
        else:
            distribution = {}
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=INVENTORY_COLUMNS).writeheader()

        num_labeled = sum(distribution.values())
        manifest["splits"][split_name] = {
            "num_active_regions": len(inventory.active_region_names),
            "num_sequences": len(inventory.sequences),
            "num_images": inventory.num_images,
            "corrupted_images_skipped": inventory.corrupted_images_skipped,
            "label_strategy": label_result.strategy_used if label_result else None,
            "label_coverage": round(label_result.coverage, 4) if label_result else 0.0,
            "num_labeled_sequences": num_labeled,
            "num_unlabeled_sequences": len(inventory.sequences) - num_labeled,
            "class_distribution": distribution,
            "label_error": label_errors.get(split_name),
            "inventory_csv": str(csv_path),
        }
        logger.info(
            "Split '%s': %d/%d sequences labeled. Class distribution: %s",
            split_name, num_labeled, len(inventory.sequences), distribution,
        )

    with (interim_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest, classes.json, and per-split inventory CSVs to %s", interim_dir)

    unresolved = [s for s, r in label_results.items() if r is None]
    if unresolved:
        logger.warning(
            "Splits with no resolved labels (excluded from class distribution, still scanned): %s",
            unresolved,
        )


if __name__ == "__main__":
    main()