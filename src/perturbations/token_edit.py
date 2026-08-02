"""Token-level edit perturbations (deletion, insertion, substitution)."""

from __future__ import annotations
import numpy as np
from typing import Optional


class TokenEditPerturber:
    """Character and token level edits for perturbation ablations.

    Used in Table in Appendix D.3 of the paper comparing perturbation types.

    Args:
        seed: Random seed.
        edit_prob: Probability of editing each token.
    """

    def __init__(self, seed: int = 2026, edit_prob: float = 0.15) -> None:
        self.seed = seed
        self.edit_prob = edit_prob
        self._rng = np.random.default_rng(seed)

    def perturb(self, text: str, mode: str = "delete") -> str:
        """Apply token-level edit.

        Args:
            text: Input text.
            mode: 'delete' | 'swap' | 'duplicate'.

        Returns:
            Perturbed text.
        """
        rng = self._rng
        words = text.split()
        if len(words) <= 2:
            return text
        if mode == "delete":
            kept = [w for w in words if rng.random() > self.edit_prob]
            return " ".join(kept) if len(kept) >= 2 else text
        elif mode == "swap":
            result = words.copy()
            for i in range(len(result) - 1):
                if rng.random() < self.edit_prob:
                    result[i], result[i + 1] = result[i + 1], result[i]
            return " ".join(result)
        elif mode == "duplicate":
            result = []
            for w in words:
                result.append(w)
                if rng.random() < self.edit_prob:
                    result.append(w)
            return " ".join(result)
        return text
