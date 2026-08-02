"""S/H ratio pre-filter for perturbation validity.

From Appendix B of the paper:
  The S/H ratio measures whether perturbations remain semantics-preserving.
  S/H > 1.0 indicates sensitivity is inflated by semantic (not surface)
  variation — the perturbation pipeline is INVALID for this domain.

  The threshold 1.0 is not empirically tuned. It follows from Definition 2.1:
  when S/H > 1.0, the perturbation has crossed the semantics-preserving
  boundary defined by the TV distance constraint.

From Table tab:real_sh of the paper:
  DistilBERT-MNLI:        S/H=0.537  -> Valid
  RoBERTa-large-MNLI:     S/H=1.373  -> Invalid (SST-2 uses classification head)
  RoBERTa-base 5% MNLI:   S/H=0.263  -> Valid
  ANLI R1:                S/H=0.649  -> Valid
  SST-2 (DistilBERT):     S/H=4.232  -> Invalid
"""

from __future__ import annotations
import numpy as np


VALIDITY_THRESHOLD = 1.0  # from Definition 2.1


class SHValidator:
    """Compute and validate the S/H ratio.

    Args:
        threshold: S/H above this value is invalid. Default 1.0.
        eps: Numerical stability.
    """

    def __init__(self, threshold: float = VALIDITY_THRESHOLD, eps: float = 1e-9) -> None:
        self.threshold = threshold
        self.eps = eps

    def sh_ratio(self, S: np.ndarray, H: np.ndarray) -> float:
        """Compute mean S/H ratio.

        Args:
            S: (N,) sensitivity array.
            H: (N,) entropy array.

        Returns:
            Mean S/H ratio.
        """
        H_safe = np.clip(H, self.eps, None)
        return float(np.mean(S / H_safe))

    def is_valid(self, S: np.ndarray, H: np.ndarray) -> bool:
        """Return True if perturbations are semantics-preserving."""
        return self.sh_ratio(S, H) < self.threshold

    def validate(self, S: np.ndarray, H: np.ndarray) -> dict:
        """Return full validity report.

        Returns:
            dict with 'sh_ratio', 'valid', 'threshold', 'message'.
        """
        ratio = self.sh_ratio(S, H)
        valid = ratio < self.threshold
        return {
            "sh_ratio": ratio,
            "valid": valid,
            "threshold": self.threshold,
            "message": (
                f"S/H={ratio:.3f} < {self.threshold} — perturbations valid."
                if valid
                else f"S/H={ratio:.3f} >= {self.threshold} — "
                     f"perturbations NOT semantics-preserving. "
                     f"SUA results unreliable for this condition."
            ),
        }
