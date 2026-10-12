"""Scores of label probabilities against true labels (scikit-learn where it
has the metric; the expected calibration error is computed here because
scikit-learn has only a binary calibration curve).

``y`` holds label indices, ``P`` one probability column per label in the
question set's label order.
"""
from __future__ import annotations

from typing import Any, Dict


def ece(y: Any, P: Any, bins: int) -> float:
    """Expected calibration error of the top label: rows binned by their top
    probability into ``bins`` equal-width bins; the size-weighted mean gap
    between each bin's accuracy and its mean top probability."""
    import numpy as np

    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def auc(y: Any, P: Any, n_labels: int) -> float:
    """ROC AUC: of the second label's probability for two labels, else the
    macro one-vs-rest mean. NaN when the rows lack a label."""
    from sklearn.metrics import roc_auc_score

    if len(set(int(v) for v in y)) < n_labels:
        return float("nan")
    if n_labels == 2:
        return float(roc_auc_score(y == 1, P[:, 1]))
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro",
                               labels=list(range(n_labels))))


def scores(y: Any, P: Any, n_labels: int, ece_bins: int) -> Dict[str, float]:
    """Accuracy of the top label, AUC, Brier score (one-class form for two
    labels, the multiclass sum otherwise), log loss and ECE."""
    from sklearn.metrics import brier_score_loss, log_loss

    labels = list(range(n_labels))
    return {
        "accuracy": float((P.argmax(axis=1) == y).mean()),
        "auc": auc(y, P, n_labels),
        "brier": float(brier_score_loss(y, P, labels=labels)),
        "log_loss": float(log_loss(y, P, labels=labels)),
        "ece": ece(y, P, ece_bins),
    }
