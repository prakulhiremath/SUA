"""Self-consistency: agreement of repeated predictions at the SAME input.

Note: Like entropy, self-consistency is a single-point diagnostic.
It measures agreement across multiple *samples* at a fixed input x,
not stability across perturbed inputs x'. It is therefore also blind
to Regime D (high sensitivity, low entropy).
"""

from __future__ import annotations
import numpy as np


class SelfConsistency:
    """Compute self-consistency score from multiple prediction samples.

    Args:
        eps: Small constant for numerical stability.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps

    def from_samples(
        self, base_preds: np.ndarray, sampled_preds: np.ndarray
    ) -> np.ndarray:
        """Compute self-consistency from base predictions and K samples.

        Args:
            base_preds: (N,) array of argmax predictions from base distribution.
            sampled_preds: (N, K) array of sampled predictions (K samples each).

        Returns:
            sc: (N,) array of agreement rates in [0, 1].
                Higher = more consistent = more confident.
        """
        N, K = sampled_preds.shape
        sc = np.zeros(N)
        for i in range(N):
            sc[i] = float(np.mean(sampled_preds[i] == base_preds[i]))
        return sc

    def from_pert_probs(
        self, base_probs: np.ndarray, pert_probs_list: list[np.ndarray]
    ) -> np.ndarray:
        """Compute SC from base probs and list of K perturbed probability arrays.

        Args:
            base_probs: (N, C) base output distributions.
            pert_probs_list: List of K arrays each (N, C).

        Returns:
            sc: (N,) self-consistency scores.
        """
        N = len(base_probs)
        K = len(pert_probs_list)
        base_preds = np.argmax(base_probs, axis=1)
        sc = np.zeros(N)
        for i in range(N):
            agreements = [
                int(np.argmax(pert_probs_list[k][i]) == base_preds[i])
                for k in range(K)
            ]
            sc[i] = float(np.mean(agreements))
        return sc
