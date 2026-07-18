"""
datasets/label_resolver.py

Implements automatic label discovery for the SDOBenchmark-style dataset.

Per project requirement, the pipeline must NEVER assume where labels come
from. Before any training happens, this module inspects the split directory
and tries, in order:

    1. A CSV metadata file (anywhere under the split root) that contains a
       column identifying each sequence and a column holding a class/flare
       label.
    2. A small per-sequence metadata file (label.json / label.txt / meta.json)
       living inside each sequence folder.
    3. A class token embedded in the sequence folder name.
    4. A class token embedded in the image filenames.

The first strategy that achieves reasonable coverage (a configurable
fraction of sequences successfully labeled) is adopted. If none succeed, a
`LabelResolutionError` is raised with actionable guidance rather than
silently guessing wrong.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from datasets.scanner import SplitInventory

logger = logging.getLogger(__name__)

# Candidate column names searched for in any CSV metadata file found.
_ID_COLUMN_CANDIDATES = ["id", "sequence", "sequence_id", "sequence_name", "name", "folder", "sample_id"]
_LABEL_COLUMN_CANDIDATES = ["label", "class", "flare_class", "goes_class", "xray_class", "target", "y", "peak_flux"]

# Recognises GOES-style flare-class tokens (A/B/C/M/X, optionally with a
# magnitude, e.g. 'M1.2', 'X5', 'C3.4') as whole, boundary-delimited tokens
# so it never matches inside an unrelated run of letters/digits (e.g. '..._XX').
_GOES_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])([ABCMX])(\d+(\.\d+)?)?(?![A-Za-z0-9])", re.IGNORECASE)
_KEYWORD_TOKENS = {
    "quiet": "QUIET", "noflare": "QUIET", "no_flare": "QUIET", "negative": "QUIET", "neg": "QUIET",
    "flare": "FLARE", "positive": "FLARE", "pos": "FLARE",
}

_MIN_COVERAGE_RATIO = 0.5  # A strategy must label at least half the sequences to be accepted.


@dataclass
class LabelResolutionResult:
    """Outcome of automatic label discovery for one split."""

    strategy_used: str
    class_names: List[str]
    label_map: Dict[str, int] = field(default_factory=dict)  # sequence_id -> class index
    coverage: float = 0.0  # fraction of scanned sequences that received a label


class LabelResolutionError(RuntimeError):
    """Raised when no labeling strategy could confidently resolve classes."""


class LabelResolver:
    """Determines how ground-truth labels are encoded for a dataset split and
    builds a sequence_id -> class_index mapping accordingly."""

    def __init__(self, config: Dict) -> None:
        """
        Args:
            config: The full parsed config.yaml dict. Reads
                `dataset.label_source`, `dataset.id_column`,
                `dataset.label_column`, and `dataset.flare_class_binning`.
        """
        self.config = config
        self.forced_source: Optional[str] = config.get("dataset", {}).get("label_source", "auto")

    def resolve(self, inventory: SplitInventory) -> LabelResolutionResult:
        """Try each labeling strategy (or only the forced one, if configured)
        and return the first one that meets the minimum coverage threshold.

        Raises:
            LabelResolutionError: If no strategy reaches sufficient coverage.
        """
        strategies = {
            "csv": self._from_csv_metadata,
            "metadata_file": self._from_metadata_files,
            "folder_name": self._from_folder_name_tokens,
            "filename": self._from_filename_tokens,
        }

        order = list(strategies.keys()) if self.forced_source in (None, "auto") else [self.forced_source]

        for name in order:
            if name not in strategies:
                raise LabelResolutionError(f"Unknown label_source '{name}' in config.yaml")
            try:
                result = strategies[name](inventory)
            except Exception as exc:  # A strategy erroring out just means "try the next one".
                logger.debug("Label strategy '%s' raised %s; trying next strategy.", name, exc)
                continue

            coverage_pct = (result.coverage * 100) if result else 0.0
            if result is not None and result.coverage >= _MIN_COVERAGE_RATIO:
                logger.info(
                    "Label source resolved as '%s' for split '%s' (coverage=%.1f%%, classes=%s).",
                    result.strategy_used, inventory.split_name, coverage_pct, result.class_names,
                )
                return result
            logger.info("Label strategy '%s' rejected for split '%s' (coverage=%.1f%%, below threshold).",
                        name, inventory.split_name, coverage_pct)

        raise LabelResolutionError(
            "Could not automatically determine labels for split "
            f"'{inventory.split_name}'. Tried: CSV metadata, per-sequence metadata "
            "files, folder-name tokens, and filename tokens. Please add a "
            "`dataset.label_source` override in config.yaml (one of: csv, "
            "metadata_file, folder_name, filename), together with any required "
            "`id_column` / `label_column` settings, or place a CSV with "
            "sequence-level labels under the split directory."
        )

    # ------------------------------------------------------------------ #
    # Strategy 1: CSV metadata file anywhere under the split root
    # ------------------------------------------------------------------ #
    def _from_csv_metadata(self, inventory: SplitInventory) -> Optional[LabelResolutionResult]:
        csv_paths = sorted(inventory.split_root.rglob("*.csv"))
        if not csv_paths:
            return None

        best_result: Optional[LabelResolutionResult] = None
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.debug("Could not parse %s as CSV: %s", csv_path, exc)
                continue

            id_col = self._find_column(df, self.config.get("dataset", {}).get("id_column"), _ID_COLUMN_CANDIDATES)
            label_col = self._find_column(df, self.config.get("dataset", {}).get("label_column"), _LABEL_COLUMN_CANDIDATES)
            if id_col is None or label_col is None:
                logger.debug("CSV %s: no usable id/label column found (id=%s, label=%s).", csv_path, id_col, label_col)
                continue

            # Build a lookup that matches on either the full id string or just
            # its basename, so IDs stored as paths ('AR/seq') or as bare
            # sequence names both resolve correctly.
            lookup: Dict[str, object] = {}
            for _, row in df.iterrows():
                raw_id = str(row[id_col]).strip().replace("\\", "/")
                lookup[raw_id.split("/")[-1]] = row[label_col]
                lookup[raw_id] = row[label_col]

            label_map: Dict[str, object] = {}
            matched = 0
            for seq in inventory.sequences:
                
                candidate_ids = [
                seq.sequence_name,
                seq.sequence_id,
                f"{seq.active_region}_{seq.sequence_name}",  # Matches meta_data.csv
                  ]
                value=None
                for candidate in candidate_ids:
                    value = lookup.get(candidate)
                    if value is not None:
                        break
                if value is None or pd.isna(value):
                     continue
                 
                
                label_map[seq.sequence_id] = value
                
                matched += 1

            coverage = matched / len(inventory.sequences) if inventory.sequences else 0.0
            if best_result is None or coverage > best_result.coverage:
                class_names, encoded_map = self._encode_labels(label_map)
                best_result = LabelResolutionResult(
                    strategy_used=f"csv:{csv_path.relative_to(inventory.split_root)}",
                    class_names=class_names, label_map=encoded_map, coverage=coverage,
                )
        return best_result

    # ------------------------------------------------------------------ #
    # Strategy 2: a small metadata file living inside each sequence folder
    # ------------------------------------------------------------------ #
    def _from_metadata_files(self, inventory: SplitInventory) -> Optional[LabelResolutionResult]:
        candidate_names = ["label.json", "label.txt", "meta.json", "metadata.json"]
        label_map: Dict[str, object] = {}
        matched = 0

        for seq in inventory.sequences:
            value = None
            for name in candidate_names:
                meta_path = seq.sequence_path / name
                if not meta_path.exists():
                    continue
                try:
                    if meta_path.suffix == ".json":
                        with meta_path.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                        value = data.get("label", data.get("class"))
                    else:
                        value = meta_path.read_text(encoding="utf-8").strip()
                except Exception as exc:
                    logger.debug("Failed reading %s: %s", meta_path, exc)
                if value is not None:
                    break
            if value is not None:
                label_map[seq.sequence_id] = value
                matched += 1

        if not inventory.sequences:
            return None
        coverage = matched / len(inventory.sequences)
        class_names, encoded_map = self._encode_labels(label_map)
        return LabelResolutionResult(strategy_used="metadata_file", class_names=class_names,
                                      label_map=encoded_map, coverage=coverage)

    # ------------------------------------------------------------------ #
    # Strategy 3: class token embedded in the sequence folder name
    # ------------------------------------------------------------------ #
    def _from_folder_name_tokens(self, inventory: SplitInventory) -> Optional[LabelResolutionResult]:
        label_map: Dict[str, object] = {}
        matched = 0
        for seq in inventory.sequences:
            token = self._extract_token(seq.sequence_name)
            if token is not None:
                label_map[seq.sequence_id] = token
                matched += 1
        if not inventory.sequences:
            return None
        coverage = matched / len(inventory.sequences)
        class_names, encoded_map = self._encode_labels(label_map)
        return LabelResolutionResult(strategy_used="folder_name", class_names=class_names,
                                      label_map=encoded_map, coverage=coverage)

    # ------------------------------------------------------------------ #
    # Strategy 4: class token embedded in image filenames (majority vote)
    # ------------------------------------------------------------------ #
    def _from_filename_tokens(self, inventory: SplitInventory) -> Optional[LabelResolutionResult]:
        label_map: Dict[str, object] = {}
        matched = 0
        for seq in inventory.sequences:
            votes: Dict[str, int] = {}
            for image_path in seq.image_paths:
                token = self._extract_token(image_path.stem)
                if token is not None:
                    votes[token] = votes.get(token, 0) + 1
            if votes:
                winner = max(votes.items(), key=lambda kv: kv[1])[0]
                label_map[seq.sequence_id] = winner
                matched += 1
        if not inventory.sequences:
            return None
        coverage = matched / len(inventory.sequences)
        class_names, encoded_map = self._encode_labels(label_map)
        return LabelResolutionResult(strategy_used="filename", class_names=class_names,
                                      label_map=encoded_map, coverage=coverage)

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_column(df: pd.DataFrame, forced: Optional[str], candidates: List[str]) -> Optional[str]:
        """Locate a usable column: an explicit override wins, otherwise try
        each candidate name case-insensitively."""
        if forced and forced in df.columns:
            return forced
        lowered = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        return None

    @staticmethod
    def _extract_token(text: str) -> Optional[str]:
        """Look for a QUIET/FLARE keyword or a GOES-class letter (A/B/C/M/X)
        as a whole token inside `text`."""
        lowered = text.lower()
        for keyword, mapped in _KEYWORD_TOKENS.items():
            if keyword in lowered:
                return mapped
        match = _GOES_TOKEN_PATTERN.search(text)
        if match:
            return match.group(1).upper()
        return None

    def _encode_labels(self, raw_label_map: Dict[str, object]) -> Tuple[List[str], Dict[str, int]]:
        """Convert arbitrary raw label values (strings, GOES classes, raw
        flux floats, etc.) into a contiguous set of integer class indices,
        applying the configured binning scheme for flare-magnitude values.
        """
        binning = self.config.get("dataset", {}).get("flare_class_binning", "multiclass")

        def normalize(value: object) -> str:
            if isinstance(value, (int, float)):
                return self._bin_peak_flux(float(value), binning)
            text = str(value).strip().upper()
            if text in ("QUIET", "FLARE"):
                return text
            goes_match = _GOES_TOKEN_PATTERN.match(text)
            if goes_match:
                return self._bin_goes_letter(goes_match.group(1).upper(), binning)
            return text  # Already a clean, custom categorical label.

        normalized = {seq_id: normalize(value) for seq_id, value in raw_label_map.items()}
        class_names = sorted(set(normalized.values()))
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        encoded_map = {seq_id: name_to_idx[name] for seq_id, name in normalized.items()}
        return class_names, encoded_map

    @staticmethod
    def _bin_goes_letter(letter: str, binning: str) -> str:
        if binning == "binary":
            return "FLARE" if letter in ("M", "X") else "QUIET"
        return letter

    @staticmethod
    def _bin_peak_flux(value: float, binning: str) -> str:
        """Bin a raw peak X-ray flux (W/m^2) into GOES-style classes using
        the standard decade thresholds, then optionally collapse to binary.
        """
        if value >= 1e-4:
            letter = "X"
        elif value >= 1e-5:
            letter = "M"
        elif value >= 1e-6:
            letter = "C"
        elif value >= 1e-7:
            letter = "B"
        else:
            letter = "A"
        return LabelResolver._bin_goes_letter(letter, binning)