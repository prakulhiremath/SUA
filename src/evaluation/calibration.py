"""Expected Calibration Error and related calibration metrics."""

from __future__ import annotations
import numpy as np


def compute_ece(
    probs: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        probs: (N, C) predicted probabilities.
        errors: (N,) binary error labels (1=error, 0=correct).
        n_bins: Number of confidence bins.

    Returns:
        ECE in [0, 1].
    """
    conf = np.max(probs, axis=1)
    correct = 1 - errors
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_b - conf_b)
    return float(ece)


def compute_ace(
    probs: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Adaptive Calibration Error (equal-frequency bins).

    Args:
        probs: (N, C) predicted probabilities.
        errors: (N,) binary error labels.
        n_bins: Number of bins.

    Returns:
        ACE in [0, 1].
    """
    conf = np.max(probs, axis=1)
    correct = 1 - errors
    n = len(conf)
    sorted_idx = np.argsort(conf)
    bin_size = n // n_bins
    ace = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else n
        idx = sorted_idx[start:end]
        if len(idx) == 0:
            continue
        acc_b = correct[idx].mean()
        conf_b = conf[idx].mean()
        ace += (len(idx) / n) * abs(acc_b - conf_b)
    return float(ace)


def calibration_curve(
    probs: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compute per-bin calibration statistics for reliability diagram.

    Returns:
        dict with 'bin_centers', 'accuracies', 'confidences', 'counts'.
    """
    conf = np.max(probs, axis=1)
    correct = 1 - errors
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        accs.append(float(correct[mask].mean()))
        confs.append(float(conf[mask].mean()))
        counts.append(int(mask.sum()))
    return {
        "bin_centers": np.array(centers),
        "accuracies": np.array(accs),
        "confidences": np.array(confs),
        "counts": np.array(counts),
    }
