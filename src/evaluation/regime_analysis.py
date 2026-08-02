"""Regime analysis: compute per-regime statistics from S and H arrays."""

from __future__ import annotations
import numpy as np
from typing import Optional
from .auroc import compute_auroc


REGIME_NAMES = {0: "A (low S, low H)", 1: "B (high S, high H)",
                2: "C (low S, high H)", 3: "D (high S, low H)"}
REGIME_SHORT = {0: "A", 1: "B", 2: "C", 3: "D"}


class RegimeAnalyzer:
    """Assign regime labels and compute per-regime diagnostics.

    The four regimes from Table 1 of the paper:
      A: low S, low H  — stable and confident (correct)
      B: high S, high H — epistemic uncertainty (correctly hedged)
      C: low S, high H  — cautious (unnecessary uncertainty)
      D: high S, low H  — MISALIGNED (the failure mode)

    Args:
        s_threshold: Sensitivity split. Default = median.
        h_threshold: Entropy split. Default = median.
    """

    def __init__(
        self,
        s_threshold: Optional[float] = None,
        h_threshold: Optional[float] = None,
    ) -> None:
        self.s_threshold = s_threshold
        self.h_threshold = h_threshold

    def fit(self, S: np.ndarray, H: np.ndarray) -> "RegimeAnalyzer":
        """Set thresholds from data."""
        self.s_threshold = float(np.median(S))
        self.h_threshold = float(np.median(H))
        return self

    def assign(self, S: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Assign regime labels (0=A, 1=B, 2=C, 3=D).

        Args:
            S: (N,) sensitivity.
            H: (N,) entropy.

        Returns:
            labels: (N,) integer array.
        """
        s_cut = self.s_threshold if self.s_threshold is not None else np.median(S)
        h_cut = self.h_threshold if self.h_threshold is not None else np.median(H)
        labels = np.zeros(len(S), dtype=int)
        labels[(S <= s_cut) & (H <= h_cut)] = 0  # A
        labels[(S >  s_cut) & (H >  h_cut)] = 1  # B
        labels[(S <= s_cut) & (H >  h_cut)] = 2  # C
        labels[(S >  s_cut) & (H <= h_cut)] = 3  # D
        return labels

    def regime_d_mask(
        self, S: np.ndarray, H: np.ndarray,
        s_pct: float = 75.0, h_pct: float = 25.0,
    ) -> np.ndarray:
        """Strict Regime D mask using percentile thresholds."""
        return (S > np.percentile(S, s_pct)) & (H < np.percentile(H, h_pct))

    def population_rate(self, S: np.ndarray, H: np.ndarray) -> dict:
        """Compute population fraction per regime."""
        labels = self.assign(S, H)
        N = len(labels)
        return {REGIME_SHORT[r]: float((labels == r).sum() / N) for r in range(4)}

    def per_regime_stats(
        self,
        S: np.ndarray,
        H: np.ndarray,
        sua: np.ndarray,
        entropy: np.ndarray,
        errors: np.ndarray,
    ) -> list[dict]:
        """Compute per-regime statistics for Table 2 (tab:regime_real).

        Returns list of dicts, one per regime.
        """
        labels = self.assign(S, H)
        rows = []
        for r in range(4):
            mask = labels == r
            n = int(mask.sum())
            if n == 0:
                continue
            err_rate = float(errors[mask].mean())
            auc_ent = compute_auroc(entropy[mask], errors[mask])
            auc_sua = compute_auroc(sua[mask], errors[mask])
            rows.append({
                "regime": REGIME_SHORT[r],
                "n": n,
                "error_rate": err_rate,
                "auroc_entropy": auc_ent,
                "auroc_sua": auc_sua,
                "delta": auc_sua - auc_ent,
            })
        return rows
