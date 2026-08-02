"""Sensitivity-Uncertainty Alignment (SUA) score.

SUA(x; epsilon, lambda) = S_theta(x; epsilon) - lambda * H_theta(Y | x)

Where:
  S_theta(x; epsilon) = E_{x' ~ Pi_eps(.|x)} [ KL( p(.|x) || p(.|x') ) ]
                      = distributional sensitivity (local output instability)
  H_theta(Y | x)      = predictive entropy (expressed uncertainty)
  lambda              = exchange-rate hyperparameter (>= 0)

SUA is POSITIVE when a model is sensitive but confident (Regime D).
SUA is NEGATIVE when a model is appropriately uncertain (Regime B/C).
SUA ~ 0 when sensitivity and expressed uncertainty are aligned (Regime A).

Critical design note (from Section 3.2 of paper):
  lambda* = 0 on standard in-distribution conditions is a DIAGNOSTIC signal,
  not a failure. It means sensitivity and entropy are already co-varied in
  well-trained models. lambda* > 0 becomes informative only when Regime D
  exists (adversarial, OOD, ambiguous inputs).
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union

from .entropy import PredictiveEntropy


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p || q) for two probability vectors.

    Args:
        p: Base distribution (C,).
        q: Perturbed distribution (C,).
        eps: Clip value to avoid log(0).

    Returns:
        KL divergence (scalar, >= 0).
    """
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    q = np.clip(q, eps, 1.0)
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


class DistributionalSensitivity:
    """Estimate S_theta(x; epsilon) via Monte Carlo KL divergence.

    S_theta(x; epsilon) = (1/K) sum_{k=1}^{K} KL( p(.|x) || p(.|x'_k) )

    where x'_1, ..., x'_K ~ Pi_eps(. | x) are semantics-preserving
    perturbations of x.

    Args:
        eps: KL divergence clip value.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps

    def __call__(
        self,
        base_probs: np.ndarray,
        pert_probs_list: list[np.ndarray],
        k_use: Optional[int] = None,
    ) -> np.ndarray:
        """Compute sensitivity for N samples using K perturbations each.

        Args:
            base_probs: (N, C) base output distributions.
            pert_probs_list: List of K arrays each (N, C).
            k_use: Use only the first k_use perturbations (for K sweep).

        Returns:
            S: (N,) sensitivity estimates.
        """
        N = len(base_probs)
        K = len(pert_probs_list) if k_use is None else min(k_use, len(pert_probs_list))
        S = np.zeros(N)
        for i in range(N):
            kls = [
                kl_divergence(base_probs[i], pert_probs_list[k][i])
                for k in range(K)
            ]
            S[i] = float(np.mean(kls))
        return S


class SUAScore:
    """Full SUA score: S_theta(x) - lambda * H_theta(Y|x).

    Args:
        lambda_val: Exchange-rate parameter. Default 1.0.
                    Set to 0.0 to use sensitivity only.
                    Selected by validation AUROC in experiments.
        eps: Numerical stability constant.
    """

    def __init__(self, lambda_val: float = 1.0, eps: float = 1e-9) -> None:
        self.lambda_val = lambda_val
        self.eps = eps
        self._sensitivity = DistributionalSensitivity(eps=eps)
        self._entropy = PredictiveEntropy(eps=eps)

    def compute(
        self,
        base_probs: np.ndarray,
        pert_probs_list: list[np.ndarray],
        k_use: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Compute SUA and its components.

        Args:
            base_probs: (N, C) base output distributions.
            pert_probs_list: List of K arrays each (N, C).
            k_use: Use only first k_use perturbations (K-sweep ablation).

        Returns:
            dict with keys:
              'sua': (N,) SUA scores
              'S':   (N,) sensitivity estimates
              'H':   (N,) entropy values
        """
        S = self._sensitivity(base_probs, pert_probs_list, k_use=k_use)
        H = self._entropy(base_probs)
        SUA = S - self.lambda_val * H
        return {"sua": SUA, "S": S, "H": H}

    def regime_labels(
        self,
        S: np.ndarray,
        H: np.ndarray,
        s_threshold: Optional[float] = None,
        h_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Assign regime labels A/B/C/D to each sample.

        Regimes (from Table 1 of paper):
          A: low S, low H  — aligned, stable, confident
          B: high S, high H — epistemic, appropriately uncertain
          C: low S, high H  — cautious, unnecessarily uncertain
          D: high S, low H  — MISALIGNED (the failure mode)

        Args:
            S: (N,) sensitivity array.
            H: (N,) entropy array.
            s_threshold: S cutoff. Default = median(S).
            h_threshold: H cutoff. Default = median(H).

        Returns:
            labels: (N,) integer array. 0=A, 1=B, 2=C, 3=D.
        """
        s_cut = np.median(S) if s_threshold is None else s_threshold
        h_cut = np.median(H) if h_threshold is None else h_threshold
        labels = np.zeros(len(S), dtype=int)
        labels[(S <= s_cut) & (H <= h_cut)] = 0  # A
        labels[(S > s_cut)  & (H > h_cut)]  = 1  # B
        labels[(S <= s_cut) & (H > h_cut)]  = 2  # C
        labels[(S > s_cut)  & (H <= h_cut)] = 3  # D
        return labels

    def regime_d_mask(
        self,
        S: np.ndarray,
        H: np.ndarray,
        s_pct: float = 75.0,
        h_pct: float = 25.0,
    ) -> np.ndarray:
        """Boolean mask for Regime D: top-s_pct% S AND bottom-h_pct% H.

        Args:
            S: (N,) sensitivity array.
            H: (N,) entropy array.
            s_pct: Sensitivity percentile threshold (default 75).
            h_pct: Entropy percentile threshold (default 25).

        Returns:
            mask: (N,) boolean array, True = Regime D.
        """
        return (S > np.percentile(S, s_pct)) & (H < np.percentile(H, h_pct))

    def sh_ratio(self, S: np.ndarray, H: np.ndarray, eps: float = 1e-9) -> float:
        """Compute mean S/H ratio as a perturbation validity pre-filter.

        From Appendix B of paper: S/H > 1.0 indicates perturbations are
        no longer semantics-preserving. The threshold 1.0 is not empirically
        tuned; it follows from Definition 2.1.

        Args:
            S: (N,) sensitivity array.
            H: (N,) entropy array.

        Returns:
            mean S/H ratio (scalar).
        """
        H_safe = np.clip(H, eps, None)
        return float(np.mean(S / H_safe))

    @classmethod
    def sweep_lambda(
        cls,
        base_probs: np.ndarray,
        pert_probs_list: list[np.ndarray],
        errors: np.ndarray,
        lambdas: list[float] = [0.0, 0.1, 0.5, 1.0, 2.0],
    ) -> dict[str, Union[float, list]]:
        """Select optimal lambda by validation AUROC.

        Args:
            base_probs: (N, C).
            pert_probs_list: List of K arrays each (N, C).
            errors: (N,) binary error labels.
            lambdas: List of lambda values to sweep.

        Returns:
            dict with 'best_lambda', 'best_auroc', 'all_aurocs'.
        """
        from sklearn.metrics import roc_auc_score

        sensitivity = DistributionalSensitivity()
        entropy_fn = PredictiveEntropy()
        S = sensitivity(base_probs, pert_probs_list)
        H = entropy_fn(base_probs)

        best_lambda = 0.0
        best_auroc = 0.0
        all_aurocs = []

        for lam in lambdas:
            sua = S - lam * H
            try:
                auc = roc_auc_score(errors, sua)
            except Exception:
                auc = 0.5
            all_aurocs.append(auc)
            if auc > best_auroc:
                best_auroc = auc
                best_lambda = lam

        return {
            "best_lambda": best_lambda,
            "best_auroc": best_auroc,
            "lambdas": lambdas,
            "aurocs": all_aurocs,
        }
