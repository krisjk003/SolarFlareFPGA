"""
scripts/predict.py

Runs inference on a single, ad-hoc observation sequence -- any folder of
chronological images, whether or not it's part of the labeled dataset (e.g.
a brand-new Active Region folder dropped in for real-time prediction).

Image discovery reuses `datasets.scanner.DatasetScanner.scan_sequence`
directly (its docstring calls this reuse out explicitly), so a folder is
scanned, corruption-checked, and chronologically ordered using exactly the
same logic as training/evaluation -- never re-implemented here.

Usage:
    python scripts/predict.py --config configs/config.yaml \\
        --input data/raw/SDOBenchmark/test/11386/2012_01_01_19_xx

    python scripts/predict.py --config configs/config.yaml \\
        --input /path/to/new_sequence --frame-mode all --output result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.scanner import DatasetScanner  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.image_utils import SUPPORTED_EXTENSIONS, preprocess_image  # noqa: E402
from utils.logger import setup_logger  # noqa: E402


def _import_build_model():
    try:
        from models import build_model  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import `build_model` from a `models` package at the project root. "
            "predict.py needs the same models/ package described in train.py's docstring "
            "(build_model(model_config: dict, num_classes: int) -> torch.nn.Module) to "
            "reconstruct the model architecture before loading checkpoint weights into it."
        ) from exc
    return build_model


def _resolve_class_names(checkpoint: Dict[str, Any], splits_dir: Path) -> List[str]:
    class_names = checkpoint.get("class_names")
    if class_names:
        return class_names
    classes_path = splits_dir / "classes.json"
    if classes_path.exists():
        with classes_path.open("r", encoding="utf-8") as f:
            return json.load(f)["classes"]
    raise KeyError(
        f"Neither the checkpoint nor {classes_path} contain a class list; cannot map predicted "
        "indices back to class names. Re-train with train.py (which stores class_names in every "
        "checkpoint) or run create_splits.py so classes.json exists."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the flare class for a single ad-hoc sequence folder.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to a folder of chronological images.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: <checkpoint_dir>/best_model.pth)")
    parser.add_argument("--frame-mode", type=str, default=None, choices=["last", "all"], help="Override dataset.frame_mode for this prediction.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the result as JSON.")
    args = parser.parse_args()

    try:
        config = load_config(Path(args.config))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: could not load config: {exc}", file=sys.stderr)
        sys.exit(1)

    paths_cfg = config.get("paths", {})
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    logger = setup_logger(Path(paths_cfg.get("log_dir", "logs")), "predict")
    device = resolve_device(training_cfg.get("device", "auto"))

    input_path = Path(args.input)
    if not input_path.is_dir():
        logger.error("--input must be a directory of images. Not found or not a directory: %s", input_path)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(paths_cfg.get("checkpoint_dir", "checkpoints")) / "best_model.pth"
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s. Train a model with train.py first, or pass --checkpoint.", checkpoint_path)
        sys.exit(1)

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    splits_dir = Path(paths_cfg.get("splits_dir", "data/splits"))
    try:
        class_names = _resolve_class_names(raw_checkpoint, splits_dir)
        build_model = _import_build_model()
    except (KeyError, ImportError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    model = build_model(config.get("model", {}), len(class_names)).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    logger.info("Loaded model from %s. Classes: %s", checkpoint_path, class_names)

    extensions = dataset_cfg.get("image_extensions", list(SUPPORTED_EXTENSIONS))
    scanner = DatasetScanner(dataset_root=input_path, image_extensions=extensions)
    record, skipped = scanner.scan_sequence(active_region=input_path.parent.name or "adhoc", seq_dir=input_path)
    if skipped:
        logger.warning("Skipped %d corrupted/unreadable image(s) in %s.", skipped, input_path)
    if not record.image_paths:
        logger.error(
            "No valid, supported images found in %s. Supported extensions: %s", input_path, extensions
        )
        sys.exit(1)

    frame_mode = args.frame_mode or dataset_cfg.get("frame_mode", "last")
    image_size = tuple(dataset_cfg.get("image_size", [224, 224]))
    mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
    std = dataset_cfg.get("std", [0.229, 0.224, 0.225])

    frame_paths = [record.image_paths[-1]] if frame_mode == "last" else list(record.image_paths)
    logger.info("Predicting on %d frame(s) (frame_mode=%s) from %s.", len(frame_paths), frame_mode, input_path)

    per_frame: List[Dict[str, Any]] = []
    prob_stack = []
    with torch.no_grad():
        for frame_path in frame_paths:
            array = preprocess_image(frame_path, image_size, mean, std)
            tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prob_stack.append(probs)
            per_frame.append({
                "image_path": str(frame_path),
                "predicted_class": class_names[int(probs.argmax())],
                "class_probabilities": {name: float(p) for name, p in zip(class_names, probs)},
            })

    mean_probs = np.mean(prob_stack, axis=0)
    predicted_index = int(mean_probs.argmax())
    result: Dict[str, Any] = {
        "input": str(input_path),
        "sequence_id": record.sequence_id,
        "frame_mode": frame_mode,
        "num_frames_used": len(frame_paths),
        "predicted_class": class_names[predicted_index],
        "predicted_index": predicted_index,
        "class_probabilities": {name: float(p) for name, p in zip(class_names, mean_probs)},
    }
    if frame_mode == "all":
        result["per_frame"] = per_frame

    logger.info("Prediction: %s (probabilities: %s)", result["predicted_class"], result["class_probabilities"])
    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved prediction to %s", output_path)


if __name__ == "__main__":
    main()