"""Synthetic four-regime benchmark (Tier 1 of the paper).

Design from Section 5.1:
  2,400 samples (600 per regime) from a Dirichlet generative model that
  independently varies S_theta and H_theta, directly instantiating Table 1.
  Ground-truth errors are assigned as a noisy sigmoid of S_theta only,
  with regime error rates 10%/30%/10%/55% for A/B/C/D.

This benchmark directly tests whether SUA detects the Regime D failure
mode by construction. It does NOT simulate any particular model's failure
distribution — it is a controlled stress test.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SyntheticDataset:
    """Container for synthetic benchmark data."""
    probs: np.ndarray          # (N, C) base distributions
    pert_probs: list           # list of K arrays each (N, C)
    errors: np.ndarray         # (N,) binary error labels
    regime_labels: np.ndarray  # (N,) 0=A, 1=B, 2=C, 3=D
    S_true: np.ndarray         # (N,) true sensitivity
    H_true: np.ndarray         # (N,) true entropy
    N_per_regime: int
    N_classes: int
    N_perturbations: int


def generate_synthetic(
    n_per_regime: int = 600,
    n_classes: int = 3,
    n_perturbations: int = 8,
    seed: int = 2026,
    error_rates: Optional[dict] = None,
    steepness: Optional[dict] = None,
) -> SyntheticDataset:
    """Generate the synthetic four-regime benchmark.

    Regime generation:
      A (low S, low H):  peaked base + tight perturbations
      B (high S, high H): uniform base + broad perturbations
      C (low S, high H):  uniform base + tight perturbations
      D (high S, low H):  peaked base + peak-shift perturbations (FAILURE MODE)

    Error assignment:
      Errors correlate with sensitivity S only (not with H or SUA).
      This is the correct leakage-free design: SUA gains predictive
      power because it contains S; entropy fails because H is nearly
      constant within Regime D.

    Args:
        n_per_regime: Samples per regime. Total = 4 * n_per_regime.
        n_classes: Output space size.
        n_perturbations: K perturbations per sample.
        seed: Random seed.
        error_rates: Dict mapping 0-3 to target error rate.
        steepness: Dict mapping 0-3 to sigmoid steepness.

    Returns:
        SyntheticDataset with all fields populated.
    """
    rng = np.random.default_rng(seed)
    C = n_classes

    if error_rates is None:
        error_rates = {0: 0.10, 1: 0.30, 2: 0.10, 3: 0.55}
    if steepness is None:
        steepness = {0: 3.0, 1: 3.5, 2: 3.0, 3: 10.0}

    # --- Base distribution generators ---
    def make_peaked(n: int) -> np.ndarray:
        probs = np.zeros((n, C))
        for i in range(n):
            dom = int(rng.integers(0, C))
            mass = float(rng.uniform(0.82, 0.96))
            rem = rng.dirichlet(np.ones(C - 1) * 0.5)
            p = np.zeros(C)
            p[dom] = mass
            others = [j for j in range(C) if j != dom]
            for k_idx, j in enumerate(others):
                p[j] = rem[k_idx] * (1 - mass)
            probs[i] = p
        return probs

    def make_uniform(n: int) -> np.ndarray:
        return rng.dirichlet(np.ones(C) * 3.0, size=n)

    # --- Perturbation generators ---
    def perturb_tight(bp: np.ndarray) -> list:
        N = len(bp)
        result = []
        for k in range(n_perturbations):
            pk = np.zeros((N, C))
            for i in range(N):
                conc = 50.0 * bp[i] + 0.5
                pk[i] = rng.dirichlet(conc)
            result.append(pk)
        return result

    def perturb_wide_uniform(bp: np.ndarray) -> list:
        N = len(bp)
        result = []
        for k in range(n_perturbations):
            pk = rng.dirichlet(np.ones(C) * 0.4, size=N)
            result.append(pk)
        return result

    def perturb_peak_shift(bp: np.ndarray) -> list:
        """Critical perturbation for Regime D: shift the dominant class."""
        N = len(bp)
        result = []
        for k in range(n_perturbations):
            pk = np.zeros((N, C))
            for i in range(N):
                dom = int(np.argmax(bp[i]))
                nd = int(rng.choice([j for j in range(C) if j != dom]))
                mass = float(rng.uniform(0.75, 0.95))
                p = np.ones(C) * ((1 - mass) / (C - 1))
                p[nd] = mass
                pk[i] = p
            result.append(pk)
        return result

    # --- Generate four regimes ---
    base_A = make_peaked(n_per_regime)
    pert_A = perturb_tight(base_A)

    base_B = make_uniform(n_per_regime)
    pert_B = perturb_wide_uniform(base_B)

    base_C = make_uniform(n_per_regime)
    pert_C = perturb_tight(base_C)

    base_D = make_peaked(n_per_regime)
    pert_D = perturb_peak_shift(base_D)   # The critical Regime D

    # --- Stack all regimes ---
    all_probs = np.concatenate([base_A, base_B, base_C, base_D], axis=0)
    all_pert = [
        np.concatenate([pert_A[k], pert_B[k], pert_C[k], pert_D[k]], axis=0)
        for k in range(n_perturbations)
    ]
    all_regimes = np.array(
        [0] * n_per_regime + [1] * n_per_regime +
        [2] * n_per_regime + [3] * n_per_regime
    )
    N_total = 4 * n_per_regime

    # --- Compute true S and H ---
    from src.metrics.sua import DistributionalSensitivity, kl_divergence
    from src.metrics.entropy import PredictiveEntropy

    S_true = np.array([
        np.mean([kl_divergence(all_probs[i], all_pert[k][i])
                 for k in range(n_perturbations)])
        for i in range(N_total)
    ])
    H_true = PredictiveEntropy()(all_probs)

    # --- Assign errors (noisy sigmoid of S only — no leakage) ---
    errors = np.zeros(N_total, dtype=int)
    err_rng = np.random.default_rng(seed + 1)

    for r in range(4):
        idx = np.where(all_regimes == r)[0]
        s_reg = S_true[idx]
        s_norm = (s_reg - s_reg.min()) / (s_reg.max() - s_reg.min() + 1e-9)
        raw_prob = 1.0 / (1.0 + np.exp(-steepness[r] * (s_norm - 0.5)))
        scale = error_rates[r] / (raw_prob.mean() + 1e-9)
        err_prob = np.clip(raw_prob * scale, 0, 1)
        draws = err_rng.random(len(idx))
        errors[idx[draws < err_prob]] = 1

    return SyntheticDataset(
        probs=all_probs,
        pert_probs=all_pert,
        errors=errors,
        regime_labels=all_regimes,
        S_true=S_true,
        H_true=H_true,
        N_per_regime=n_per_regime,
        N_classes=n_classes,
        N_perturbations=n_perturbations,
    )
