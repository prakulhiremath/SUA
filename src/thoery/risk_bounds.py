"""Structural bounds from Theorems 4.1 and 4.2 of the paper.

These provide formal grounding for SUA as a diagnostic, NOT operational
guarantees. The kappa(x) term (interpretation-collapse gap) is unobservable
in general. The bounds are informative in the qualitative direction they imply.

Theorem 4.1 (SUA in worst-case perturbed risk):
  R_rob(x) <= R_theta(x) + L_D * SUA(x) + kappa(x)

Theorem 4.2 (SUA violation and calibration error):
  ECE >= (1/B) sum_b (S_bar_b - lambda * H_bar_b - c_b)+

Proposition 4.3 (Selective risk control):
  R_sel(tau) <= R_theta + L_D * tau + E[kappa(x) | SUA(x) <= tau]
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def sua_risk_bound(
    sua_scores: np.ndarray,
    base_risk: float,
    L_D: float = 1.0,
    kappa: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-sample worst-case risk upper bound (Theorem 4.1).

    R_rob(x) <= R_theta(x) + L_D * SUA(x; eps, lambda) + kappa(x)

    Args:
        sua_scores: (N,) SUA scores.
        base_risk: Scalar baseline risk R_theta(x).
        L_D: Lipschitz constant of the loss (Assumption A1).
        kappa: (N,) interpretation-collapse gap. If None, assumed 0
               (kappa=0 gives the tightest version of the bound).

    Returns:
        bound: (N,) upper bounds on worst-case perturbed risk.
    """
    bound = base_risk + L_D * sua_scores
    if kappa is not None:
        bound = bound + kappa
    return np.maximum(bound, 0.0)  # risk is non-negative


def calibration_lower_bound(
    S_bar: np.ndarray,
    H_bar: np.ndarray,
    lambda_val: float = 1.0,
    c_b: Optional[np.ndarray] = None,
) -> float:
    """Compute ECE lower bound from SUA violation (Theorem 4.2).

    ECE >= (1/B) sum_b (S_bar_b - lambda * H_bar_b - c_b)+

    The positive-part operator fires ONLY in Regime D (S_bar > lambda * H_bar).
    This identifies the structural origin of miscalibration.

    Args:
        S_bar: (B,) mean sensitivity per calibration bin.
        H_bar: (B,) mean entropy per calibration bin.
        lambda_val: Exchange-rate parameter.
        c_b: (B,) bin-level kappa averages. If None, assumed 0.

    Returns:
        Scalar ECE lower bound.
    """
    B = len(S_bar)
    mismatch = S_bar - lambda_val * H_bar
    if c_b is not None:
        mismatch = mismatch - c_b
    positive_part = np.maximum(mismatch, 0.0)
    return float(positive_part.mean())


def selective_risk_bound(
    sua_scores: np.ndarray,
    tau: float,
    base_risk: float,
    L_D: float = 1.0,
) -> float:
    """Compute selective risk bound for abstention threshold tau (Prop 4.3).

    R_sel(tau) <= R_theta + L_D * tau + E[kappa | SUA <= tau]

    (kappa term set to 0 for the computable bound.)

    Args:
        sua_scores: (N,) SUA scores.
        tau: Abstention threshold (predict if SUA <= tau).
        base_risk: Scalar baseline risk.
        L_D: Lipschitz constant.

    Returns:
        Upper bound on selective risk.
    """
    return base_risk + L_D * tau
