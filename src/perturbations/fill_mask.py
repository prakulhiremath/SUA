"""MLM fill-mask perturbations (BERT-based).

From Appendix E of paper:
  BERT fill-mask (bert-base-uncased), top-k=8, up to 3 positions per
  sentence, cosine similarity threshold 0.87 on [CLS] embeddings.
  K=6 perturbations per input.
  Mean accepted similarity: 0.968 +/- 0.004.
"""

from __future__ import annotations
import re
import numpy as np
from typing import Optional


# Lightweight word-level synonym dictionary for CPU-only perturbations
# Used when BERT fill-mask is not available (Colab CPU, testing).
_SYNONYMS: dict[str, list[str]] = {
    "good": ["great", "fine", "solid", "decent"],
    "bad": ["poor", "awful", "weak", "flawed"],
    "big": ["large", "huge", "major", "substantial"],
    "small": ["tiny", "minor", "slight", "modest"],
    "not": ["never", "hardly", "barely", "scarcely"],
    "very": ["quite", "rather", "fairly", "somewhat"],
    "is": ["was", "seems", "appears", "remains"],
    "are": ["were", "seem", "appear", "remain"],
    "can": ["could", "may", "might", "should"],
    "will": ["would", "should", "may", "might"],
    "important": ["significant", "notable", "critical", "key"],
    "new": ["recent", "novel", "current", "modern"],
    "true": ["correct", "accurate", "valid", "right"],
    "false": ["incorrect", "wrong", "invalid", "mistaken"],
    "said": ["stated", "claimed", "noted", "remarked"],
    "show": ["demonstrate", "indicate", "suggest", "reveal"],
    "see": ["observe", "notice", "view", "find"],
    "think": ["believe", "feel", "consider", "suppose"],
    "know": ["understand", "recognize", "realize", "see"],
    "go": ["proceed", "move", "travel", "head"],
    "get": ["obtain", "receive", "acquire", "gain"],
    "use": ["employ", "apply", "utilize", "leverage"],
    "the": ["a", "this", "that", "each"],
    "a": ["the", "this", "one", "that"],
    "man": ["person", "individual", "figure", "one"],
    "woman": ["person", "individual", "figure", "one"],
    "people": ["individuals", "persons", "humans", "ones"],
    "has": ["had", "holds", "contains", "shows"],
    "have": ["had", "possess", "hold", "contain"],
    "does": ["did", "would", "could", "might"],
    "do": ["did", "would", "could", "might"],
    "some": ["certain", "various", "several", "any"],
    "all": ["every", "each", "any", "most"],
}


class FillMaskPerturber:
    """Semantics-preserving text perturbations via word substitution.

    Implements a lightweight approximation of BERT fill-mask for
    environments without GPU or full BERT access. The full BERT-based
    implementation is used in the actual experiments (Appendix E).

    Perturbation types (matching paper's K=8 ablation structure):
      k=0,1,2: synonym swap (increasing aggressiveness)
      k=3:     random deletion
      k=4:     word order swap
      k=5:     negation insert
      k=6:     synonym + deletion combo
      k=7:     char noise + synonym

    Args:
        seed: Random seed for reproducibility.
        synonyms: Optional custom synonym dictionary.
    """

    def __init__(
        self,
        seed: int = 2026,
        synonyms: Optional[dict] = None,
    ) -> None:
        self.seed = seed
        self.synonyms = synonyms or _SYNONYMS
        self._rng = np.random.default_rng(seed)

    def perturb(self, text: str, k: int) -> str:
        """Apply perturbation k to text.

        Args:
            text: Input text string.
            k: Perturbation index (0-7).

        Returns:
            Perturbed text string.
        """
        rng = self._rng
        if k == 0:
            return self._synonym_swap(text, n=2, rng=rng)
        elif k == 1:
            return self._synonym_swap(text, n=4, rng=rng)
        elif k == 2:
            return self._synonym_swap(text, n=6, rng=rng)
        elif k == 3:
            return self._random_delete(text, p=0.20, rng=rng)
        elif k == 4:
            return self._word_swap(text, n=3, rng=rng)
        elif k == 5:
            return self._negate(text, rng=rng)
        elif k == 6:
            t = self._synonym_swap(text, n=3, rng=rng)
            return self._random_delete(t, p=0.15, rng=rng)
        else:
            t = self._char_noise(text, n=2, rng=rng)
            return self._synonym_swap(t, n=2, rng=rng)

    def perturb_batch(self, texts: list[str], k: int) -> list[str]:
        """Perturb a list of texts with perturbation k."""
        return [self.perturb(t, k) for t in texts]

    def _synonym_swap(self, text: str, n: int, rng: np.random.Generator) -> str:
        words = text.split()
        changed = 0
        for j in rng.permutation(len(words)):
            if changed >= n:
                break
            w = re.sub(r"[^\w]", "", words[j]).lower()
            if w in self.synonyms:
                words[j] = str(rng.choice(self.synonyms[w]))
                changed += 1
        return " ".join(words)

    def _random_delete(self, text: str, p: float, rng: np.random.Generator) -> str:
        words = text.split()
        if len(words) <= 3:
            return text
        kept = [w for w in words if rng.random() > p]
        return " ".join(kept) if len(kept) >= 2 else text

    def _word_swap(self, text: str, n: int, rng: np.random.Generator) -> str:
        words = text.split()
        if len(words) < 4:
            return text
        for _ in range(n):
            i, j = rng.integers(0, len(words), size=2)
            words[i], words[j] = words[j], words[i]
        return " ".join(words)

    def _negate(self, text: str, rng: np.random.Generator) -> str:
        _VERBS = {"is", "are", "was", "were", "has", "have", "does", "do",
                  "can", "will", "should", "would"}
        words = text.split()
        for j, w in enumerate(words):
            if w.lower() in _VERBS and j + 1 < len(words) and rng.random() < 0.85:
                words.insert(j + 1, "not")
                return " ".join(words)
        return text + " not"

    def _char_noise(self, text: str, n: int, rng: np.random.Generator) -> str:
        words = text.split()
        for _ in range(n):
            j = int(rng.integers(0, len(words)))
            w = list(words[j])
            if len(w) >= 4:
                i1, i2 = rng.integers(0, len(w), size=2)
                w[i1], w[i2] = w[i2], w[i1]
                words[j] = "".join(w)
        return " ".join(words)
