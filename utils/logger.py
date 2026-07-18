"""
utils/logger.py

Central logging configuration so every script (inspect_dataset.py, train.py,
evaluate.py, predict.py) writes to both the console and a shared, timestamped
log file with a consistent format.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(log_dir: Path, name: str) -> logging.Logger:
    """Configure the root logger with a timestamped file handler under
    `log_dir` plus a console handler, then return a named logger for the
    calling script.

    Args:
        log_dir: Directory where the log file will be written (created if
            it does not exist).
        name: Name used both for the log file prefix and the returned
            logger (e.g. "train", "evaluate").

    Returns:
        A configured `logging.Logger` instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()  # Avoid duplicate handlers if setup is called more than once.

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(name)
    logger.info("Logging initialised. Log file: %s", log_file)
    return logger