"""
datasets/sdo_dataset.py

PyTorch Dataset wrapping the scanned SDOBenchmark sequences. Each element is
a single, fully preprocessed image tensor (resized, 3-channel, normalised)
paired with an integer class label.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.scanner import SequenceRecord
from utils.image_utils import preprocess_image

logger = logging.getLogger(__name__)


class SDOBenchmarkDataset(Dataset):
    """Turns a list of scanned `SequenceRecord`s plus a resolved label map
    into a flat, indexable PyTorch dataset.

    Two frame-selection modes are supported (configurable via
    `dataset.frame_mode` in config.yaml):

        "last": one sample per sequence, using the most recent chronological
            image. This matches a single-image 2D CNN architecture and the
            way real-time flare prediction consumes "current state".
        "all": one sample per valid image, with every image in a sequence
            sharing that sequence's label. Useful for data-hungry training
            or later temporal / incremental-learning extensions.
    """

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        label_map: Dict[str, int],
        image_size: Tuple[int, int],
        mean: Sequence[float],
        std: Sequence[float],
        frame_mode: str = "last",
    ) -> None:
        """
        Args:
            records: Sequences discovered by DatasetScanner.
            label_map: sequence_id -> integer class index, from LabelResolver.
            image_size: (width, height) target size for every image.
            mean, std: Per-channel normalisation statistics.
            frame_mode: "last" or "all" (see class docstring).

        Raises:
            ValueError: If the resulting dataset would be empty.
        """
        self.image_size = tuple(image_size)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.samples: List[Tuple[str, int]] = self._build_index(records, label_map, frame_mode)
        if not self.samples:
            raise ValueError(
                "SDOBenchmarkDataset was built with zero samples; check that "
                "label_map covers these sequences and that frame_mode is valid."
            )

    @staticmethod
    def _build_index(
        records: Sequence[SequenceRecord], label_map: Dict[str, int], frame_mode: str
    ) -> List[Tuple[str, int]]:
        """Flatten sequences + labels into a list of (image_path, label) pairs."""
        index: List[Tuple[str, int]] = []
        for record in records:
            if record.sequence_id not in label_map or not record.image_paths:
                continue  # Unlabeled or empty sequences are silently excluded (already logged upstream).
            label = label_map[record.sequence_id]
            if frame_mode == "last":
                index.append((str(record.image_paths[-1]), label))
            elif frame_mode == "all":
                index.extend((str(p), label) for p in record.image_paths)
            else:
                raise ValueError(f"Unknown frame_mode '{frame_mode}'. Use 'last' or 'all'.")
        return index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load, preprocess, and return one (image_tensor, label) pair.

        Images that fail to decode here (despite passing the lightweight
        scan-time check) are treated as corrupted: logged and replaced with
        a neutral zero tensor rather than crashing the whole training run.
        """
        path_str, label = self.samples[idx]
        try:
            array = preprocess_image(path_str, self.image_size, self.mean, self.std)
        except Exception as exc:
            logger.warning("Failed to load image at __getitem__ (%s): %s. Using a blank tensor instead.",
                            path_str, exc)
            array = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.float32)

        # HWC -> CHW for PyTorch.
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()
        return tensor, label