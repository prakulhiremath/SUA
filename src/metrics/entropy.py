"""Predictive entropy: H(Y|x) = -sum_y p(y|x) log p(y|x)."""

from __future__ import annotations
import numpy as np
from typing import Union


class PredictiveEntropy:
    """Compute Shannon entropy of a probability distribution.

    This is the standard single-point uncertainty metric.
    It is BLIND to local sensitivity: two distributions can have
    identical entropy but completely different sensitivity to input
    perturbations (the core limitation motivating SUA).

    Args:
        eps: Small constant to avoid log(0). Default 1e-9.
        base: Log base. None = natural log (nats). 2 = bits.
    """

    def __init__(self, eps: float = 1e-9, base: Union[float, None] = None) -> None:
        self.eps = eps
        self.base = base

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        """Compute entropy for a batch of distributions.

        Args:
            probs: Array of shape (N, C) — N distributions over C classes.
                   Each row must sum to ~1.

        Returns:
            entropies: Array of shape (N,) — entropy per sample.
        """
        probs = np.clip(probs, self.eps, 1.0)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        H = -np.sum(probs * np.log(probs), axis=-1)
        if self.base is not None:
            H = H / np.log(self.base)
        return H

    def scalar(self, p: np.ndarray) -> float:
        """Entropy of a single distribution p (shape: (C,))."""
        p = np.clip(p, self.eps, 1.0)
        p = p / p.sum()
        h = float(-np.sum(p * np.log(p)))
        if self.base is not None:
            h = h / np.log(self.base)
        return h
