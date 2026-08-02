"""AUROC computation for failure prediction."""

from __future__ import annotations
import numpy as np
from typing import Optional
from sklearn.metrics import roc_auc_score


def compute_auroc(scores: np.ndarray, errors: np.ndarray) -> float:
    """Compute AUROC for failure prediction.

    Higher score = predicted as more likely to fail.

    Args:
        scores: (N,) uncertainty/risk scores.
        errors: (N,) binary error labels (1=error, 0=correct).

    Returns:
        AUROC in [0, 1]. Returns 0.5 if undefined (single class).
    """
    if errors.sum() == 0 or errors.sum() == len(errors):
        return float("nan")
    try:
        return float(roc_auc_score(errors, scores))
    except Exception:
        return 0.5


def compute_regime_auroc(
    scores: np.ndarray,
    errors: np.ndarray,
    regime_labels: np.ndarray,
    n_regimes: int = 4,
) -> dict[str, float]:
    """Compute per-regime AUROC.

    Args:
        scores: (N,) uncertainty scores.
        errors: (N,) binary error labels.
        regime_labels: (N,) integer regime assignments (0=A,1=B,2=C,3=D).
        n_regimes: Number of regimes.

    Returns:
        dict mapping regime name to AUROC.
    """
    regime_names = {0: "A", 1: "B", 2: "C", 3: "D"}
    results = {}
    for r in range(n_regimes):
        mask = regime_labels == r
        if mask.sum() == 0:
            results[regime_names[r]] = float("nan")
            continue
        results[regime_names[r]] = compute_auroc(scores[mask], errors[mask])
    return results


def compute_delta_auroc(
    scores_sua: np.ndarray,
    scores_entropy: np.ndarray,
    errors: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute AUROC(SUA) - AUROC(Entropy), optionally on a subset.

    Args:
        scores_sua: (N,) SUA scores.
        scores_entropy: (N,) entropy scores.
        errors: (N,) binary error labels.
        mask: Optional boolean mask for subset evaluation.

    Returns:
        Delta AUROC (SUA advantage).
    """
    if mask is not None:
        scores_sua = scores_sua[mask]
        scores_entropy = scores_entropy[mask]
        errors = errors[mask]
    return compute_auroc(scores_sua, errors) - compute_auroc(scores_entropy, errors)
