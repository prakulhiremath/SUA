"""Selective prediction / abstention evaluation."""

from __future__ import annotations
import numpy as np
from typing import Union


def selective_accuracy(
    scores: np.ndarray,
    errors: np.ndarray,
    coverage: float = 0.80,
) -> float:
    """Accuracy on the lowest-score (most reliable) fraction.

    Abstention rule: abstain if score > threshold, where threshold
    is set so that exactly `coverage` fraction of samples are kept.

    Args:
        scores: (N,) uncertainty scores. Lower = more reliable.
        errors: (N,) binary error labels.
        coverage: Fraction of samples to keep.

    Returns:
        Accuracy on kept samples.
    """
    n_keep = max(1, int(len(scores) * coverage))
    keep_idx = np.argsort(scores)[:n_keep]
    return float(1.0 - errors[keep_idx].mean())


def coverage_curve(
    scores: np.ndarray,
    errors: np.ndarray,
    n_points: int = 30,
) -> dict:
    """Compute selective accuracy across a range of coverages.

    Args:
        scores: (N,) uncertainty scores.
        errors: (N,) binary error labels.
        n_points: Number of coverage levels.

    Returns:
        dict with 'coverages' and 'accuracies'.
    """
    coverages = np.linspace(0.1, 1.0, n_points)
    accs = [selective_accuracy(scores, errors, c) for c in coverages]
    return {"coverages": coverages, "accuracies": np.array(accs)}


def high_confidence_error_rate(
    probs: np.ndarray,
    errors: np.ndarray,
    confidence_threshold: float = 0.85,
) -> float:
    """Fraction of errors that occur at high confidence.

    High-confidence errors are the most dangerous: they are acted upon
    without review. SUA's abstention rule targets these specifically.

    Args:
        probs: (N, C) predicted probabilities.
        errors: (N,) binary error labels.
        confidence_threshold: Confidence level above which errors are 'high-confidence'.

    Returns:
        Fraction of predictions that are both high-confidence and wrong.
    """
    conf = np.max(probs, axis=1)
    hc_mask = conf >= confidence_threshold
    if hc_mask.sum() == 0:
        return 0.0
    return float(errors[hc_mask].mean())


def hce_reduction(
    probs: np.ndarray,
    sua_scores: np.ndarray,
    errors: np.ndarray,
    coverage: float = 0.90,
    confidence_threshold: float = 0.85,
) -> float:
    """High-confidence error reduction at given coverage via SUA abstention.

    Computes (HCE_baseline - HCE_SUA) / HCE_baseline.

    Args:
        probs: (N, C) predicted probabilities.
        sua_scores: (N,) SUA scores.
        errors: (N,) binary error labels.
        coverage: Fraction of samples kept.
        confidence_threshold: Confidence threshold for HCE.

    Returns:
        Relative HCE reduction (e.g., 0.31 = 31% reduction).
    """
    baseline_hce = high_confidence_error_rate(probs, errors, confidence_threshold)
    if baseline_hce == 0.0:
        return 0.0
    n_keep = max(1, int(len(sua_scores) * coverage))
    keep_idx = np.argsort(sua_scores)[:n_keep]
    sua_hce = high_confidence_error_rate(probs[keep_idx], errors[keep_idx], confidence_threshold)
    return float((baseline_hce - sua_hce) / baseline_hce)
