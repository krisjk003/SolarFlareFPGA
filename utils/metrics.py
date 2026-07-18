"""
utils/metrics.py

Classification metric computation shared by evaluation.Evaluator. Kept
separate from plotting (utils/visualization.py) so numeric results can be
inspected or serialised independently of matplotlib.
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


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute the full metric suite required by the project spec: accuracy,
    macro precision/recall/F1, confusion matrix, a textual classification
    report, and (for binary problems, when `y_prob` is supplied) ROC-AUC
    plus ROC curve points.

    Args:
        y_true: Ground-truth integer class labels.
        y_pred: Predicted integer class labels.
        y_prob: Optional (N, num_classes) predicted probabilities, required
            to compute ROC-AUC / ROC curve for binary problems.
        class_names: Optional human-readable class names, in index order.

    Returns:
        Dict containing all computed metrics.
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

    return results