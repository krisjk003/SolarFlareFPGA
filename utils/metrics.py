"""
utils/metrics.py

Classification metric computation shared by evaluation.Evaluator and, as of
this update, train.py's Trainer (per-epoch validation). Kept separate from
plotting (utils/visualization.py) so numeric results can be inspected or
serialised independently of matplotlib.

In addition to the original accuracy / macro precision-recall-F1 /
confusion matrix / classification report / ROC-AUC suite, this module now
also derives FLARE-specific precision, recall, F1, and TSS (True Skill
Statistic) directly from the confusion matrix, whenever `class_names`
describes the project's binary FLARE/QUIET scheme. These are computed from
predicted *labels* (not probabilities), so they're available both here
(evaluate.py, where `y_prob` is also known) and from train.py's per-epoch
validation pass, which only ever has argmax predictions. They're additive:
every key the original function returned is still returned, unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _find_class_index(class_names: Optional[List[str]], target_name: str) -> Optional[int]:
    """Best-effort, case-insensitive lookup of `target_name` in `class_names`.

    Returns None (rather than raising) if `class_names` is missing or
    doesn't contain it, since compute_classification_metrics is called from
    contexts (e.g. train.py's per-epoch validation) that should simply skip
    the FLARE-specific metrics below rather than error out.

    Note: this intentionally duplicates the small lookup evaluate.py already
    does for itself (`_find_flare_index`) rather than importing it -- utils/
    must not depend on scripts/, and this file needs its own copy either way
    since it's called from train.py too, not just evaluate.py.
    """
    if not class_names:
        return None
    for idx, name in enumerate(class_names):
        if name.strip().upper() == target_name.upper():
            return idx
    return None


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute the full metric suite required by the project spec: accuracy,
    macro precision/recall/F1, confusion matrix, a textual classification
    report, and (for binary problems, when `y_prob` is supplied) ROC-AUC
    plus ROC curve points. Additionally, for the project's binary
    FLARE/QUIET scheme, also returns FLARE-specific precision/recall/F1 and
    TSS (True Skill Statistic), derived from the confusion matrix alone.

    Args:
        y_true: Ground-truth integer class labels.
        y_pred: Predicted integer class labels.
        y_prob: Optional (N, num_classes) predicted probabilities, required
            to compute ROC-AUC / ROC curve for binary problems.
        class_names: Optional human-readable class names, in index order.

    Returns:
        Dict containing all computed metrics. `flare_precision`,
        `flare_recall`, `flare_f1`, and `tss` are only present when a class
        literally named "FLARE" is found among `class_names` and there are
        exactly two classes -- they're omitted (not zero-filled) otherwise,
        so callers can tell "not applicable" from "zero score".
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    results: Dict = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "precision_macro": precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
        "classification_report": classification_report(
            y_true_arr, y_pred_arr, target_names=class_names, zero_division=0
        ),
    }

    num_classes = len(class_names) if class_names else int(max(y_true_arr.max(), y_pred_arr.max()) + 1)
    if num_classes == 2 and y_prob is not None:
        # Positive-class probability is column index 1 by convention.
        positive_scores = y_prob[:, 1]
        results["roc_auc"] = roc_auc_score(y_true_arr, positive_scores)
        fpr, tpr, _ = roc_curve(y_true_arr, positive_scores)
        results["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # ------------------------------------------------------------------
    # FLARE-specific precision/recall/F1/TSS (new).
    # ------------------------------------------------------------------
    # Used by train.py's Trainer to pick the best checkpoint on a signal
    # that actually reflects minority-class performance, instead of
    # accuracy/loss (which a collapsed "always predict QUIET" model can
    # already win). Recomputed with an explicit `labels=[0, 1]` confusion
    # matrix (separate from the one stored above, which is left exactly as
    # before) so indexing by class index is always safe even if one class
    # happens to be entirely absent from this particular y_true/y_pred.
    flare_index = _find_class_index(class_names, "FLARE")
    if flare_index is not None and num_classes == 2:
        other_index = 1 - flare_index
        flare_cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
        tp = int(flare_cm[flare_index, flare_index])
        fn = int(flare_cm[flare_index, other_index])
        fp = int(flare_cm[other_index, flare_index])
        tn = int(flare_cm[other_index, other_index])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # a.k.a. TPR
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results["flare_precision"] = precision
        results["flare_recall"] = recall
        results["flare_f1"] = f1
        results["tss"] = recall - fpr_rate  # True Skill Statistic

    return results