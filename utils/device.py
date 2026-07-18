"""
utils/device.py

Single implementation of the 'auto | cuda | cpu' device-resolution logic,
shared by train.py, evaluate.py, and predict.py so the rule lives in exactly
one place.
"""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device preference string from config.yaml into a
    torch.device.

    Args:
        preference: One of "auto", "cuda", or "cpu".

    Returns:
        The resolved torch.device.

    Raises:
        RuntimeError: If "cuda" is explicitly requested but unavailable.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested device='cuda' but no CUDA device is available.")
        return torch.device("cuda")
    # "auto": prefer GPU, fall back to CPU transparently.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")