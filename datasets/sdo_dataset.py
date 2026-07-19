"""
datasets/sdo_dataset.py

PyTorch Dataset wrapping the scanned SDOBenchmark sequences. Each element is
a single, deterministically preprocessed image (resized, 3-channel) paired
with an integer class label. Loading/resizing is delegated entirely to
`utils.image_utils.preprocess_image`, which never normalises or augments --
that happens here, in `__getitem__`, via an injectable torchvision
`transform` (e.g. `train.py`'s `train_transform` with augmentations, or
`val_transform` without).

If no `transform` is supplied, the dataset falls back to a plain
ToTensor() + Normalize(mean, std) pipeline built from the `mean`/`std`
arguments, so existing callers that construct this dataset without an
explicit transform (e.g. evaluate.py) keep getting correctly normalised,
unaugmented tensors exactly as before this refactor.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            records: Sequences discovered by DatasetScanner.
            label_map: sequence_id -> integer class index, from LabelResolver.
            image_size: (width, height) target size for every image.
            mean, std: Per-channel normalisation statistics. Used to build
                the fallback transform below when `transform` is omitted.
            frame_mode: "last" or "all" (see class docstring).
            transform: Optional torchvision-style callable applied to each
                (PIL) image in `__getitem__`, expected to end in ToTensor()
                (and typically Normalize()) so it returns a CHW float tensor.
                Pass augmentations here for training (see `train.py`'s
                `train_transform`) or a plain ToTensor()+Normalize() for
                validation/inference (`val_transform`). If omitted, a
                ToTensor()+Normalize(mean, std) pipeline is used by default.

        Raises:
            ValueError: If the resulting dataset would be empty.
        """
        self.image_size = tuple(image_size)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.transform = transform
        # Backward-compatible default: callers that don't pass an explicit
        # `transform` (e.g. evaluate.py) still get correctly normalised,
        # unaugmented tensors, matching this dataset's pre-refactor behaviour.
        self._default_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )
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
        a neutral zero tensor -- built directly, bypassing `transform` --
        rather than crashing the whole training run.
        """
        path_str, label = self.samples[idx]
        try:
            array = preprocess_image(path_str, self.image_size)
        except Exception as exc:
            logger.warning("Failed to load image at __getitem__ (%s): %s. Using a blank tensor instead.",
                            path_str, exc)
            array = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.float32)
            # HWC -> CHW for PyTorch.
            return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float(), label

        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        transform = self.transform if self.transform is not None else self._default_transform
        tensor = transform(image)
        return tensor, label