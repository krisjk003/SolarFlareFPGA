"""
utils/config.py

Loads and validates the project's single YAML configuration file so every
hyperparameter lives in one place (configs/config.yaml) instead of being
scattered or hardcoded across scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# Every script depends on these top-level sections existing; fail fast and
# loudly if the config file is malformed rather than crashing deep inside
# some later function with a confusing KeyError.
REQUIRED_TOP_LEVEL_KEYS = ["dataset", "model", "training", "paths"]


def load_config(config_path: Path) -> Dict[str, Any]:
    """Read a YAML config file into a plain nested dict and sanity-check
    that the sections the rest of the pipeline depends on are present.

    Args:
        config_path: Path to a YAML file (e.g. configs/config.yaml).

    Returns:
        The parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file does not parse into a dict, or is missing
            required top-level sections.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file did not parse into a dictionary: {config_path}")

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config file {config_path} is missing required sections: {missing}")

    return config


def get_nested(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Convenience accessor for nested config values.

    Example:
        get_nested(config, "dataset.image_size", [224, 224])
    """
    node: Any = config
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node