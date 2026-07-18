"""
utils/image_utils.py

Image I/O helpers shared across the dataset scanner, the PyTorch Dataset,
and predict.py. Centralised here so loading/resizing/normalisation logic is
implemented exactly once (avoids duplicate code across modules).

Supports JPG/JPEG/PNG via Pillow and FITS via astropy. FITS support is
optional at import time: if astropy is missing, only an informative error is
raised when a .fits file is actually encountered, so JPG/PNG-only datasets
keep working without the extra dependency.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence, Tuple, Union
import random

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

try:
    from astropy.io import fits
    _ASTROPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _ASTROPY_AVAILABLE = False

PathLike = Union[str, os.PathLike]

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".fits", ".fit")



def is_supported_image(path: PathLike, extensions: Sequence[str] = SUPPORTED_EXTENSIONS) -> bool:
    """Return True if `path` has one of the supported image extensions
    (case-insensitive)."""
    suffix = Path(path).suffix.lower()
    return suffix in tuple(ext.lower() for ext in extensions)


def quick_verify_image(path: PathLike) -> bool:
    """Cheap structural check used while scanning the dataset. Does not do a
    full pixel decode (that happens lazily in `load_image_as_array`), so
    scanning thousands of files stays fast. Returns False for anything that
    looks corrupted, empty, or unreadable — never raises.
    """
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return False
        if path.suffix.lower() in (".fits", ".fit"):
            if not _ASTROPY_AVAILABLE:
                logger.warning(
                    "Found a .fits file but astropy is not installed "
                    "(pip install astropy). Skipping: %s", path,
                )
                return False
            with fits.open(path, memmap=False) as hdul:
                _ = hdul[0].data  # Force a read to catch truncated/corrupt FITS files.
            return True
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:  # Broad catch is intentional: any failure means "unusable image".
        return False


def load_image_as_array(path: PathLike) -> np.ndarray:
    """Load a single image (jpg/jpeg/png/fits) into a float32 numpy array in
    the range [0, 255].

    Returns:
        (H, W) array for grayscale/FITS sources, or (H, W, 3) for RGB
        sources. Callers should pass the result through `to_three_channel`.

    Raises:
        RuntimeError / ValueError / OSError: On any read/decode failure —
        callers are expected to catch and treat the sample as corrupted.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".fits", ".fit"):
        if not _ASTROPY_AVAILABLE:
            raise RuntimeError("astropy is required to read FITS files. Install with `pip install astropy`.")
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
        if data is None:
            raise ValueError(f"FITS file has no image data in its primary HDU: {path}")
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            # Some FITS cubes store multiple frames/channels; use the first plane.
            data = np.squeeze(data)
            if data.ndim != 2:
                data = data[0]

        # FITS solar images commonly have very large/variable dynamic range
        # (raw DN counts). Robustly rescale to [0, 255] using percentile
        # clipping so a handful of hot/cosmic-ray pixels don't wash out the
        # rest of the frame.
        lo, hi = np.percentile(data, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1.0
        data = np.clip((data - lo) / (hi - lo), 0.0, 1.0) * 255.0
        return data.astype(np.float32)  # (H, W)

    with Image.open(path) as img:
        img = img.convert("RGB")
        array = np.asarray(img, dtype=np.float32)
    return array  # (H, W, 3)


def to_three_channel(image: np.ndarray) -> np.ndarray:
    """Ensure an image array has 3 channels, replicating grayscale data if
    necessary (requirement: convert grayscale images into 3 channels).

    Accepts (H, W), (H, W, 1), or (H, W, 3) arrays.
    """
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected image shape for channel conversion: {image.shape}")
    return image


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an (H, W, 3) float32 array to `size` = (width, height) using
    PIL's high-quality Lanczos filter.
    """
    img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    img = img.resize(size, resample=Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def normalize_image(image: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Scale pixels to [0, 1] then apply per-channel standardisation."""
    image = image / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    return (image - mean_arr) / std_arr


def preprocess_image(
    path: PathLike, size: Tuple[int, int], mean: Sequence[float], std: Sequence[float]
) -> np.ndarray:
    """Full preprocessing pipeline for a single image file: load -> RGB ->
    resize -> normalise. Returns an (H, W, 3) float32 array ready to be
    converted to a CHW torch tensor by the caller.
    """
    array = load_image_as_array(path)
    array = to_three_channel(array)
    array = resize_image(array, size)
    array = normalize_image(array, mean, std)
    return array