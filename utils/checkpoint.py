"""
utils/checkpoint.py

Helpers for saving/loading model checkpoints, shared by training.Trainer,
evaluate.py, and predict.py so there is exactly one implementation of the
checkpoint file format (avoids duplicate code / format drift).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Path,
    last_name: str = "last_model.pth",
    best_name: str = "best_model.pth",
) -> None:
    """Always overwrite `last_model.pth` with the current state; additionally
    copy it to `best_model.pth` whenever `is_best` is True.

    Args:
        state: Dict to persist (model/optimizer/scaler state, epoch, best
            metric, history, config, etc.).
        is_best: Whether this epoch is the best seen so far.
        checkpoint_dir: Directory to write checkpoints into (created if
            missing).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / last_name
    torch.save(state, last_path)
    if is_best:
        best_path = checkpoint_dir / best_name
        shutil.copyfile(last_path, best_path)
        logger.info("New best model saved to %s (epoch %s).", best_path, state.get("epoch"))


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into `model` (and optionally optimizer/scaler for
    resuming training).

    Args:
        path: Path to a .pth checkpoint file.
        model: Model instance to load weights into (architecture must
            already match, e.g. same num_classes).
        optimizer: Optional optimizer to restore state into (for resume).
        scaler: Optional AMP GradScaler to restore state into (for resume).

    Returns:
        The raw checkpoint dict (epoch, history, config, etc.) for further use.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If the checkpoint is missing the expected 'model_state' key.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model_state"])
    except KeyError as exc:
        raise KeyError(f"Checkpoint at {path} is missing 'model_state'.") from exc

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    logger.info("Loaded checkpoint from %s (epoch %s).", path, checkpoint.get("epoch"))
    return checkpoint