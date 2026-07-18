"""
scripts/model.py

Implements the one contract train.py, evaluate.py, and predict.py all
require and none of the uploaded folders provided:

    build_model(model_config: dict, num_classes: int) -> torch.nn.Module

`model_config` is the `model:` section of config.yaml, e.g.:

    model:
      name: resnet18          # architecture identifier, see _MODEL_FAMILIES below
      pretrained: true        # use ImageNet-pretrained torchvision weights
      dropout: 0.2            # optional, dropout applied right before the new head
      freeze_backbone: false  # optional, train only the new classifier head

Input/output contract (see datasets.sdo_dataset.SDOBenchmarkDataset, which
is the only thing that ever calls a model built here): every model receives
(N, 3, H, W) float tensors -- already resized and normalised -- and must
return (N, num_classes) raw logits. Softmax/argmax happen in the calling
script (Trainer.validate, Evaluator.predict, predict.py's main), never
inside the model itself.

Supported `model.name` values:
    resnet18, resnet34, resnet50, resnet101        (torchvision, ImageNet head)
    mobilenet_v2, efficientnet_b0, efficientnet_b1, vgg16   (torchvision)
    simple_cnn                                       (small from-scratch CNN,
                                                        no torchvision/internet
                                                        dependency -- useful for
                                                        offline smoke tests)

NOTE ON PROJECT LAYOUT: train.py/evaluate.py/predict.py all import this via
`from models import build_model`, i.e. a package literally named `models`
at the project root (a sibling of datasets/, utils/, scripts/) -- that
package was never uploaded. Point a `models/__init__.py` at this file, e.g.:

    # models/__init__.py
    from scripts.model import build_model  # noqa: F401

or simply copy this file's contents to `models/__init__.py`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

# name -> (torchvision constructor name, torchvision Weights enum name).
# Kept as plain strings and looked up with getattr() so this module only
# imports torchvision lazily, inside the functions that actually need it --
# `simple_cnn` (and just inspecting this file) never requires torchvision.
_RESNET_FAMILY: Dict[str, Tuple[str, str]] = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "resnet101": ("resnet101", "ResNet101_Weights"),
}

# Torchvision architectures whose classification head is a `.classifier`
# Sequential ending in a Linear layer, rather than resnet's single `.fc`.
_CLASSIFIER_FAMILY: Dict[str, Tuple[str, str]] = {
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "efficientnet_b1": ("efficientnet_b1", "EfficientNet_B1_Weights"),
    "vgg16": ("vgg16", "VGG16_Weights"),
}

_SUPPORTED_NAMES = sorted(list(_RESNET_FAMILY) + list(_CLASSIFIER_FAMILY) + ["simple_cnn"])


def build_model(model_config: Dict[str, Any], num_classes: int) -> nn.Module:
    """Construct the classification model described by `model_config`.

    Args:
        model_config: The `model:` section of config.yaml. Reads `name`
            (default "resnet18"), `pretrained` (default True), `dropout`
            (default 0.0), and `freeze_backbone` (default False).
        num_classes: Number of output classes, resolved from the dataset
            (see preprocess.py / create_splits.py's classes.json) -- never
            hardcoded by the caller.

    Returns:
        A torch.nn.Module mapping (N, 3, H, W) input to (N, num_classes)
        logits.

    Raises:
        ValueError: If `num_classes` is invalid, or `model.name` is not one
            of the architectures this module knows how to build.
    """
    if num_classes is None or num_classes < 2:
        raise ValueError(f"num_classes must be an integer >= 2, got {num_classes!r}.")

    name = str(model_config.get("name", "resnet18")).strip().lower()
    pretrained = bool(model_config.get("pretrained", True))
    dropout = float(model_config.get("dropout", 0.0))
    freeze_backbone = bool(model_config.get("freeze_backbone", False))

    if name in _RESNET_FAMILY:
        model, head = _build_resnet_family(name, pretrained, num_classes, dropout)
    elif name in _CLASSIFIER_FAMILY:
        model, head = _build_classifier_family(name, pretrained, num_classes, dropout)
    elif name == "simple_cnn":
        model, head = _build_simple_cnn(num_classes, dropout)
    else:
        raise ValueError(f"Unknown model.name '{name}' in config.yaml. Supported values: {_SUPPORTED_NAMES}")

    if freeze_backbone:
        _freeze_all_but(model, head)
        logger.info("model.freeze_backbone=true: backbone frozen, only the new classifier head is trainable.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built model '%s' (num_classes=%d, pretrained=%s, dropout=%.2f): %d/%d trainable params.",
        name, num_classes, pretrained, dropout, trainable, total,
    )
    return model


def _replace_head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    """A fresh classifier head: Linear, optionally preceded by Dropout."""
    if dropout > 0:
        return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def _build_resnet_family(name: str, pretrained: bool, num_classes: int, dropout: float) -> Tuple[nn.Module, nn.Module]:
    """resnet18/34/50/101: single `.fc` Linear head."""
    from torchvision import models as tv_models

    ctor_name, weights_enum_name = _RESNET_FAMILY[name]
    constructor: Callable[..., nn.Module] = getattr(tv_models, ctor_name)
    weights = getattr(tv_models, weights_enum_name).DEFAULT if pretrained else None

    model = constructor(weights=weights)
    in_features = model.fc.in_features
    model.fc = _replace_head(in_features, num_classes, dropout)
    return model, model.fc


def _build_classifier_family(name: str, pretrained: bool, num_classes: int, dropout: float) -> Tuple[nn.Module, nn.Module]:
    """mobilenet_v2 / efficientnet_b0 / efficientnet_b1 / vgg16: `.classifier`
    Sequential ending in a Linear layer. The whole `.classifier` block is
    replaced with a single fresh head so `dropout` from config.yaml is
    applied consistently regardless of architecture."""
    from torchvision import models as tv_models

    ctor_name, weights_enum_name = _CLASSIFIER_FAMILY[name]
    constructor: Callable[..., nn.Module] = getattr(tv_models, ctor_name)
    weights = getattr(tv_models, weights_enum_name).DEFAULT if pretrained else None

    model = constructor(weights=weights)
    last_linear = next(m for m in reversed(model.classifier) if isinstance(m, nn.Linear))
    model.classifier = _replace_head(last_linear.in_features, num_classes, dropout)
    return model, model.classifier


def _conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _build_simple_cnn(num_classes: int, dropout: float) -> Tuple[nn.Module, nn.Module]:
    """A small from-scratch CNN with no torchvision/internet dependency.
    Four conv blocks with increasing width, global average pooling (so any
    input resolution works, not just 224x224), then a linear head. Useful
    for `model.pretrained: false`-style offline runs, quick pipeline smoke
    tests, or datasets too small to fine-tune a large pretrained backbone.
    """
    features = nn.Sequential(
        _conv_block(3, 32),
        nn.MaxPool2d(2),
        _conv_block(32, 64),
        nn.MaxPool2d(2),
        _conv_block(64, 128),
        nn.MaxPool2d(2),
        _conv_block(128, 256),
        nn.AdaptiveAvgPool2d(1),
    )
    head = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout), nn.Linear(256, num_classes))
    model = nn.Sequential(features, head)
    return model, head


def _freeze_all_but(model: nn.Module, trainable_module: nn.Module) -> None:
    """Freeze every parameter in `model` except those belonging to
    `trainable_module` (the newly-created classifier head)."""
    trainable_param_ids = {id(p) for p in trainable_module.parameters()}
    for param in model.parameters():
        param.requires_grad = id(param) in trainable_param_ids