"""
datasets/scanner.py

Recursively scans the SDOBenchmark-style dataset directory tree and builds
an in-memory inventory of Active Regions, Observation Sequences, and the
(chronologically ordered, corruption-filtered) image files inside each
sequence.

No folder names are ever assumed or hardcoded: every Active Region and every
Sequence directory name is discovered dynamically at run time, so the code
works unchanged regardless of how many Active Regions or sequences exist.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

from utils.image_utils import is_supported_image, quick_verify_image

logger = logging.getLogger(__name__)

# Pulls every run of digits out of a filename so chronologically-named files
# (timestamps, frame indices, etc.) sort in true time order rather than
# lexicographic order (e.g. so '...9.jpg' sorts before '...10.jpg').
_DIGIT_PATTERN = re.compile(r"\d+")


def _natural_sort_key(path: Path) -> Tuple[int, ...]:
    """Build a numeric sort key from every run of digits in a filename."""
    digits = _DIGIT_PATTERN.findall(path.stem)
    if not digits:
        return (0,)
    return tuple(int(d) for d in digits)


@dataclass
class SequenceRecord:
    """One observation sequence: a folder of chronologically-ordered images
    belonging to a single Active Region."""

    active_region: str
    sequence_name: str
    sequence_path: Path
    image_paths: List[Path] = field(default_factory=list)

    @property
    def sequence_id(self) -> str:
        """Stable, human-readable identifier used for label matching."""
        return f"{self.active_region}/{self.sequence_name}"


@dataclass
class SplitInventory:
    """Full inventory of one dataset split (e.g. 'training' or 'test')."""

    split_name: str
    split_root: Path
    sequences: List[SequenceRecord] = field(default_factory=list)
    corrupted_images_skipped: int = 0

    @property
    def active_region_names(self) -> List[str]:
        """Sorted, de-duplicated list of every Active Region name found."""
        return sorted({seq.active_region for seq in self.sequences})

    @property
    def num_images(self) -> int:
        """Total number of valid images across every sequence in this split."""
        return sum(len(seq.image_paths) for seq in self.sequences)


class DatasetScanner:
    """Discovers Active Region / Sequence / Image structure on disk without
    ever assuming fixed folder names or a fixed number of images/sequences.
    """

    def __init__(self, dataset_root: Path, image_extensions: Sequence[str]) -> None:
        """
        Args:
            dataset_root: Directory containing the split folders
                (e.g. .../SDOBenchmark/, which contains training/ and test/).
            image_extensions: Accepted file extensions, e.g. from config.yaml.
        """
        self.dataset_root = Path(dataset_root)
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

    def scan_split(self, split_name: str) -> SplitInventory:
        """Recursively scan `dataset_root/split_name` and return a full
        inventory of every Active Region / Sequence / image discovered.

        Directory convention assumed only structurally (never by name):
            split_root/<ActiveRegion>/<Sequence>/<images...>

        Raises:
            FileNotFoundError: If the split directory does not exist.
        """
        split_root = self.dataset_root / split_name
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        inventory = SplitInventory(split_name=split_name, split_root=split_root)

        active_region_dirs = sorted(p for p in split_root.iterdir() if p.is_dir())
        for ar_dir in active_region_dirs:
            try:
                sequence_dirs = sorted(p for p in ar_dir.iterdir() if p.is_dir())
            except OSError as exc:
                logger.warning("Could not read Active Region directory %s: %s", ar_dir, exc)
                continue

            for seq_dir in sequence_dirs:
                record, skipped = self.scan_sequence(ar_dir.name, seq_dir)
                inventory.corrupted_images_skipped += skipped
                if record.image_paths:  # Sequences left with zero valid images are excluded entirely.
                    inventory.sequences.append(record)
                else:
                    logger.warning("Sequence has no usable images, excluding: %s", seq_dir)

        logger.info(
            "Scanned split '%s': %d active regions, %d sequences, %d images "
            "(%d corrupted images skipped).",
            split_name, len(inventory.active_region_names), len(inventory.sequences),
            inventory.num_images, inventory.corrupted_images_skipped,
        )
        return inventory

    def scan_sequence(self, active_region: str, seq_dir: Path) -> Tuple[SequenceRecord, int]:
        """Scan a single sequence folder: keep only readable, supported
        images, sorted into chronological order.

        Public (not a private helper) so predict.py can reuse the exact same
        loading/validation/ordering logic for a single ad-hoc sequence.

        Returns:
            (SequenceRecord, number_of_corrupted_images_skipped)
        """
        try:
            candidate_files = [
                p for p in seq_dir.iterdir()
                if p.is_file() and is_supported_image(p, self.image_extensions)
            ]
        except OSError as exc:
            logger.warning("Could not read sequence directory %s: %s", seq_dir, exc)
            return SequenceRecord(active_region, seq_dir.name, seq_dir, []), 0

        candidate_files.sort(key=lambda p: (_natural_sort_key(p), p.name))

        valid_files: List[Path] = []
        skipped = 0
        for image_path in candidate_files:
            if quick_verify_image(image_path):
                valid_files.append(image_path)
            else:
                skipped += 1
                logger.warning("Skipping corrupted/unreadable image: %s", image_path)

        record = SequenceRecord(
            active_region=active_region,
            sequence_name=seq_dir.name,
            sequence_path=seq_dir,
            image_paths=valid_files,
        )
        return record, skipped