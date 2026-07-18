"""
scripts/utils.py

Small, script-level orchestration helpers shared by every CLI entry point
under scripts/ (clean_dataset.py, preprocess.py, create_splits.py, train.py,
evaluate.py, predict.py).

This is deliberately separate from the project's `utils` package (utils/,
from utils.zip): that package holds generic, reusable infrastructure (image
I/O, checkpoints, metrics, plotting, device/config primitives). This module
holds the thinner CLI-wiring patterns that were copy-pasted near-identically
into every script's main() instead of being shared:

    * loading config.yaml and exiting with a consistent message on failure
      (present, near-verbatim, in all six scripts)
    * resolving `dataset.image_extensions` (duplicated in preprocess.py and
      clean_dataset.py, inlined again in train.py/evaluate.py/predict.py)
    * discovering split subdirectories under `dataset.root` (duplicated in
      preprocess.py and clean_dataset.py)
    * resolving a --checkpoint CLI argument, defaulting to
      `<checkpoint_dir>/best_model.pth` (duplicated in evaluate.py and
      predict.py)

None of these change behaviour versus the code they replace -- they just
give the duplicated logic exactly one home.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the project root (parent of this scripts/ folder) importable so this
# module works whether it's imported by another script or run/tested
# directly -- same pattern used by every other file under scripts/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import load_config  # noqa: E402
from utils.image_utils import SUPPORTED_EXTENSIONS  # noqa: E402


def load_config_or_exit(config_path: str) -> Dict[str, Any]:
    """Load config.yaml, printing a consistent error to stderr and exiting
    the process (status 1) on failure.

    This mirrors what every script's main() did inline: it has to report to
    stderr rather than through the logger, because log_dir itself comes
    from the config that just failed to load.
    """
    try:
        return load_config(Path(config_path))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: could not load config: {exc}", file=sys.stderr)
        sys.exit(1)


def get_image_extensions(config: Dict[str, Any]) -> List[str]:
    """Resolve `dataset.image_extensions` from a parsed config.yaml,
    defaulting to every extension `utils.image_utils` knows how to read."""
    return config.get("dataset", {}).get("image_extensions", list(SUPPORTED_EXTENSIONS))


def discover_splits(dataset_root: Path) -> List[str]:
    """Every immediate subdirectory of `dataset_root` is treated as a split
    (e.g. training/test) -- nothing about split names is ever hardcoded,
    per the project's "don't assume folder names" rule.

    Raises:
        FileNotFoundError: If `dataset_root` does not exist.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"dataset.root '{dataset_root}' does not exist. Update configs/config.yaml, "
            "or check that the raw dataset was extracted to this path."
        )
    return sorted(p.name for p in dataset_root.iterdir() if p.is_dir())


def resolve_checkpoint_path(checkpoint_arg: Optional[str], paths_cfg: Dict[str, Any], logger: Any) -> Path:
    """Resolve the --checkpoint CLI argument shared by evaluate.py and
    predict.py: an explicit path if given, otherwise
    `<paths.checkpoint_dir>/best_model.pth`.

    Exits the process (status 1) if nothing exists at the resolved path --
    both callers need a real checkpoint to do anything useful, so this
    mirrors their existing fail-fast behaviour rather than raising and
    pushing the sys.exit() back onto every caller.
    """
    checkpoint_path = (
        Path(checkpoint_arg) if checkpoint_arg
        else Path(paths_cfg.get("checkpoint_dir", "checkpoints")) / "best_model.pth"
    )
    if not checkpoint_path.exists():
        logger.error(
            "Checkpoint not found: %s. Train a model with train.py first, or pass --checkpoint.",
            checkpoint_path,
        )
        sys.exit(1)
    return checkpoint_path