"""SPUQ baseline: Perturbation-Based Uncertainty Quantification.

Gao et al. (2024). SPUQ: Perturbation-Based Uncertainty Quantification
for Large Language Models. EACL 2024.

SPUQ measures uncertainty as the variance of predictions across
semantics-preserving perturbations. Unlike SUA, it does not condition
on whether the model already expresses appropriate uncertainty through
its entropy — it treats all sensitivity as uncertainty.
"""

from __future__ import annotations
import numpy as np


class SPUQ:
    """SPUQ: variance-based uncertainty from perturbations.

    Key difference from SUA:
      SPUQ = Var[p(y|x')] across x' ~ Pi_eps(.|x)
      SUA  = E[KL(p(.|x) || p(.|x'))] - lambda * H(Y|x)

    SPUQ does not subtract entropy, so it cannot distinguish:
      - Regime B (high sensitivity + high entropy): correct uncertainty
      - Regime D (high sensitivity + low entropy): failure mode

    Args:
        reduction: How to aggregate variance across classes.
                   'mean': mean variance over classes.
                   'max': max variance over classes.
                   'trace': trace of covariance.
    """

    def __init__(self, reduction: str = "mean") -> None:
        assert reduction in ("mean", "max", "trace")
        self.reduction = reduction

    def __call__(
        self,
        base_probs: np.ndarray,
        pert_probs_list: list[np.ndarray],
    ) -> np.ndarray:
        """Compute SPUQ uncertainty estimates.

        Args:
            base_probs: (N, C) base output distributions.
            pert_probs_list: List of K arrays each (N, C).

        Returns:
            uncertainty: (N,) SPUQ uncertainty scores.
        """
        N, C = base_probs.shape
        K = len(pert_probs_list)
        # Stack: (N, K, C)
        stack = np.stack(pert_probs_list, axis=1)
        # Variance across K perturbations for each class: (N, C)
        var = np.var(stack, axis=1)

        if self.reduction == "mean":
            return var.mean(axis=1)
        elif self.reduction == "max":
            return var.max(axis=1)
        else:  # trace
            return var.sum(axis=1)
