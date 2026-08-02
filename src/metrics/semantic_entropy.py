"""Semantic Entropy baseline (Kuhn et al., 2023).

Measures semantic variability across multiple generated outputs
for a fixed input. Complementary to SUA (which measures sensitivity
to INPUT perturbations, not output variability).

This implementation provides a lightweight approximation for
comparison purposes in classification settings.
"""

from __future__ import annotations
import numpy as np
from .entropy import PredictiveEntropy


class SemanticEntropy:
    """Approximate semantic entropy for classification tasks.

    In classification, semantic equivalence classes correspond to label
    categories. We compute entropy over the marginal label distribution
    averaged across K samples, which approximates semantic entropy
    when label identity implies semantic equivalence.

    Args:
        eps: Numerical stability constant.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps
        self._entropy = PredictiveEntropy(eps=eps)

    def from_pert_probs(
        self, base_probs: np.ndarray, pert_probs_list: list[np.ndarray]
    ) -> np.ndarray:
        """Compute semantic entropy from base + perturbed distributions.

        Averages the distributions across K+1 draws (base + K perturbed)
        and computes entropy of the mixture. This captures variability
        across the semantic neighborhood rather than at a single point.

        Args:
            base_probs: (N, C).
            pert_probs_list: List of K arrays each (N, C).

        Returns:
            se: (N,) semantic entropy estimates.
        """
        N, C = base_probs.shape
        K = len(pert_probs_list)
        # Stack all K+1 distributions
        stack = np.stack([base_probs] + pert_probs_list, axis=1)  # (N, K+1, C)
        # Mixture distribution (equal weights)
        mixture = stack.mean(axis=1)  # (N, C)
        return self._entropy(mixture)
