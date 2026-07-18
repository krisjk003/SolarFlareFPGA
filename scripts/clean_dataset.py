"""
scripts/clean_dataset.py

CLI entry point that audits the raw SDOBenchmark-style dataset tree for
integrity problems *before* any preprocessing or training happens:

    * corrupted / unreadable image files
    * sequence folders that end up with zero usable images
    * Active Region folders that end up with zero usable sequences
    * whether ground-truth labels can be automatically resolved at all,
      and if not, exactly where the pipeline expected to find them

Nothing about the folder layout is hardcoded: Active Regions, sequences,
and even the split names themselves (e.g. "training" / "test") are all
discovered dynamically from whatever subdirectories actually exist under
`dataset.root`, using `datasets.scanner.DatasetScanner` and
`datasets.label_resolver.LabelResolver` (the exact same classes used by
preprocess.py, so the audit reflects reality, not a re-implementation).

By default this script only *reports* problems (dry run). Pass --apply to
actually move corrupted image files into a quarantine folder, mirroring
their original relative path -- files are never permanently deleted, and
directory structure / metadata files are never touched.

Usage:
    python scripts/clean_dataset.py --config configs/config.yaml
    python scripts/clean_dataset.py --config configs/config.yaml --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make the project root (parent of this scripts/ folder) importable so this
# file works when run directly, e.g. `python scripts/clean_dataset.py`,
# regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.label_resolver import LabelResolutionError, LabelResolver  # noqa: E402
from datasets.scanner import DatasetScanner  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.image_utils import SUPPORTED_EXTENSIONS, is_supported_image, quick_verify_image  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

_MAX_LISTED_ITEMS = 200  # Cap long lists in the saved report for readability.


def _get_image_extensions(config: Dict[str, Any]) -> List[str]:
    return config.get("dataset", {}).get("image_extensions", list(SUPPORTED_EXTENSIONS))


def _discover_splits(dataset_root: Path) -> List[str]:
    """Every immediate subdirectory of `dataset_root` is treated as a split
    (e.g. training/test), whatever it happens to be named."""
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"dataset.root '{dataset_root}' does not exist. Update configs/config.yaml "
            "or check that the datasets/ upload was extracted to this path."
        )
    return sorted(p.name for p in dataset_root.iterdir() if p.is_dir())


def _audit_sequence(seq_dir: Path, extensions: List[str]) -> Tuple[int, List[Path], int]:
    """Inspect one sequence folder directly on disk (independent of the
    scanner's in-memory inventory) so the report can name exact bad files.

    Returns:
        (num_valid_images, corrupted_image_paths, non_image_file_count)
    """
    try:
        entries = [p for p in seq_dir.iterdir() if p.is_file()]
    except OSError:
        return 0, [], 0

    image_candidates = [p for p in entries if is_supported_image(p, extensions)]
    non_image_count = len(entries) - len(image_candidates)

    corrupted: List[Path] = [p for p in image_candidates if not quick_verify_image(p)]
    valid_count = len(image_candidates) - len(corrupted)
    return valid_count, corrupted, non_image_count


def _audit_split(split_root: Path, extensions: List[str]) -> Dict[str, Any]:
    """Walk one split (AR -> sequence -> images) and collect every integrity
    issue found, without assuming any folder names."""
    active_regions = sorted(p for p in split_root.iterdir() if p.is_dir())

    total_sequences = 0
    images_valid = 0
    images_corrupted = 0
    non_image_files = 0
    corrupted_files: List[str] = []
    empty_sequences: List[str] = []
    empty_active_regions: List[str] = []

    for ar_dir in active_regions:
        seq_dirs = sorted(p for p in ar_dir.iterdir() if p.is_dir())
        ar_has_usable_sequence = False

        for seq_dir in seq_dirs:
            total_sequences += 1
            valid_count, corrupted, non_image_count = _audit_sequence(seq_dir, extensions)
            images_valid += valid_count
            images_corrupted += len(corrupted)
            non_image_files += non_image_count
            corrupted_files.extend(str(p) for p in corrupted)

            if valid_count == 0:
                empty_sequences.append(str(seq_dir))
            else:
                ar_has_usable_sequence = True

        if not seq_dirs or not ar_has_usable_sequence:
            empty_active_regions.append(str(ar_dir))

    truncated = len(corrupted_files) > _MAX_LISTED_ITEMS
    return {
        "active_regions": len(active_regions),
        "sequences": total_sequences,
        "images_valid": images_valid,
        "images_corrupted": images_corrupted,
        "non_image_files_count": non_image_files,
        "corrupted_files": corrupted_files[:_MAX_LISTED_ITEMS],
        "corrupted_files_truncated": truncated,
        "empty_sequences": empty_sequences[:_MAX_LISTED_ITEMS],
        "empty_sequences_truncated": len(empty_sequences) > _MAX_LISTED_ITEMS,
        "empty_active_regions": empty_active_regions,
        "_all_corrupted_files": corrupted_files,  # kept internally for --apply; stripped before saving
    }


def _check_labels(split_name: str, split_root: Path, extensions: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Reuse the real LabelResolver (never re-implemented here) purely to
    report, per the project requirement, exactly where labels are expected
    if they cannot be found automatically."""
    scanner = DatasetScanner(split_root.parent, extensions)
    try:
        inventory = scanner.scan_split(split_name)
        result = LabelResolver(config).resolve(inventory)
        return {
            "resolved": True,
            "strategy_used": result.strategy_used,
            "class_names": result.class_names,
            "coverage": round(result.coverage, 4),
            "error": None,
        }
    except LabelResolutionError as exc:
        return {"resolved": False, "strategy_used": None, "class_names": [], "coverage": 0.0, "error": str(exc)}


def _apply_quarantine(corrupted_files: List[str], dataset_root: Path, quarantine_dir: Path, logger) -> int:
    """Move (never delete) each corrupted file into `quarantine_dir`,
    preserving its path relative to dataset_root. Returns count moved."""
    import shutil

    moved = 0
    for file_str in corrupted_files:
        src = Path(file_str)
        try:
            relative = src.relative_to(dataset_root)
        except ValueError:
            relative = Path(src.name)
        dest = quarantine_dir / relative
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved += 1
        except OSError as exc:
            logger.warning("Could not quarantine %s: %s", src, exc)
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit (and optionally clean) the raw SDOBenchmark dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually move corrupted image files into the quarantine folder. Default is a dry run (report only).",
    )
    parser.add_argument("--report", type=str, default=None, help="Override output path for the JSON report.")
    args = parser.parse_args()

    try:
        config = load_config(Path(args.config))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: could not load config: {exc}", file=sys.stderr)
        sys.exit(1)

    paths_cfg = config.get("paths", {})
    logger = setup_logger(Path(paths_cfg.get("log_dir", "logs")), "clean_dataset")

    dataset_root = Path(config["dataset"]["root"])
    extensions = _get_image_extensions(config)

    try:
        splits = _discover_splits(dataset_root)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if not splits:
        logger.error("No split subdirectories found under %s. Nothing to audit.", dataset_root)
        sys.exit(1)

    logger.info("Discovered %d split(s) under %s: %s", len(splits), dataset_root, splits)

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "splits": {},
        "quarantine_applied": bool(args.apply),
        "quarantine_dir": None,
    }

    all_corrupted: List[str] = []
    for split_name in splits:
        split_root = dataset_root / split_name
        logger.info("Auditing split '%s' ...", split_name)
        split_report = _audit_split(split_root, extensions)
        all_corrupted.extend(split_report.pop("_all_corrupted_files"))
        split_report["label_check"] = _check_labels(split_name, split_root, extensions, config)
        report["splits"][split_name] = split_report

        logger.info(
            "Split '%s': %d active regions, %d sequences, %d valid images, %d corrupted images, "
            "%d empty sequences, %d empty active regions.",
            split_name, split_report["active_regions"], split_report["sequences"],
            split_report["images_valid"], split_report["images_corrupted"],
            len(split_report["empty_sequences"]), len(split_report["empty_active_regions"]),
        )
        if split_report["label_check"]["resolved"]:
            logger.info(
                "Split '%s': labels resolved via '%s' (coverage=%.1f%%, classes=%s).",
                split_name, split_report["label_check"]["strategy_used"],
                split_report["label_check"]["coverage"] * 100, split_report["label_check"]["class_names"],
            )
        else:
            logger.warning(
                "Split '%s': labels could NOT be automatically resolved. Details: %s",
                split_name, split_report["label_check"]["error"],
            )

    if args.apply and all_corrupted:
        quarantine_dir = Path(paths_cfg.get("quarantine_dir", "data/quarantine"))
        logger.info("Quarantining %d corrupted file(s) into %s ...", len(all_corrupted), quarantine_dir)
        moved = _apply_quarantine(all_corrupted, dataset_root, quarantine_dir, logger)
        report["quarantine_dir"] = str(quarantine_dir)
        logger.info("Quarantined %d/%d corrupted file(s).", moved, len(all_corrupted))
    elif args.apply:
        logger.info("No corrupted files found; nothing to quarantine.")

    report_path = Path(args.report) if args.report else Path(paths_cfg.get("report_dir", "reports")) / (
        f"clean_dataset_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written to %s", report_path)

    any_labels_missing = any(not s["label_check"]["resolved"] for s in report["splits"].values())
    if any_labels_missing:
        logger.warning(
            "One or more splits have no automatically resolvable labels. See the 'label_check' "
            "section of the report above for exactly what was tried and what is expected."
        )


if __name__ == "__main__":
    main()