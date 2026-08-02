"""Model wrappers for consistent probability extraction across tasks.

All wrappers return numpy arrays of shape (N, C) so the same SUA
computation applies regardless of task or architecture.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


def _apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Softmax with temperature scaling."""
    t = torch.tensor(logits / T, dtype=torch.float32)
    return F.softmax(t, dim=-1).numpy()


class ClassificationModelWrapper:
    """Wrap a HuggingFace sequence classification model.

    Handles label remapping (critical for models where id2label order
    differs from dataset label order — e.g., textattack/bert-base-uncased-snli).

    Args:
        model_name: HuggingFace model identifier.
        label_remap: Optional dict {dataset_label: model_output_class}.
                     If None, identity mapping is used.
        temperature: Temperature scaling for distribution spreading.
                     T > 1 spreads distributions (needed for confident models
                     to have non-trivial sensitivity signal).
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_name: str,
        label_remap: Optional[dict] = None,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.label_remap = label_remap
        self.temperature = temperature
        self.device = device
        self._tok = None
        self._mdl = None

    def load(self) -> "ClassificationModelWrapper":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._mdl = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self._mdl.eval()

        # Auto-detect label mapping from config if available
        if self.label_remap is None:
            id2label = self._mdl.config.id2label
            named = {v.lower(): k for k, v in id2label.items()}
            if all(k in named for k in ("entailment", "neutral", "contradiction")):
                # NLI model with named labels
                # SNLI: 0=entailment, 1=neutral, 2=contradiction
                self.label_remap = {
                    0: named["entailment"],
                    1: named["neutral"],
                    2: named["contradiction"],
                }
        return self

    def get_logits(self, text_a: str, text_b: Optional[str] = None) -> np.ndarray:
        """Get raw logits for a single example."""
        if text_b:
            inp = self._tok(text_a, text_b, return_tensors="pt",
                            truncation=True, max_length=128, padding=True).to(self.device)
        else:
            inp = self._tok(text_a, return_tensors="pt",
                            truncation=True, max_length=128, padding=True).to(self.device)
        with torch.no_grad():
            return self._mdl(**inp).logits.cpu().numpy()[0]

    def get_probs(self, text_a: str, text_b: Optional[str] = None) -> np.ndarray:
        """Get temperature-scaled probabilities."""
        return _apply_temperature(self.get_logits(text_a, text_b), self.temperature)

    def get_probs_batch(
        self, texts_a: list[str], texts_b: Optional[list[str]] = None
    ) -> np.ndarray:
        """Get probs for a batch. Returns (N, C)."""
        N = len(texts_a)
        probs = []
        for i in range(N):
            tb = texts_b[i] if texts_b else None
            probs.append(self.get_probs(texts_a[i], tb))
        return np.array(probs)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Argmax prediction, applying label_remap if set."""
        raw_preds = np.argmax(probs, axis=-1)
        return raw_preds  # remap handled at dataset level

    def remap_labels(self, dataset_labels: np.ndarray) -> np.ndarray:
        """Remap dataset label integers to model output class integers."""
        if self.label_remap is None:
            return dataset_labels
        return np.array([self.label_remap[int(l)] for l in dataset_labels])


class NLIModelWrapper(ClassificationModelWrapper):
    """NLI-specific wrapper with automatic label detection."""

    KNOWN_MODELS = {
        "cross-encoder/nli-MiniLM2-L6-H768": {
            0: 1, 1: 2, 2: 0  # SNLI: ent=1, neut=2, cont=0
        },
        "textattack/distilbert-base-uncased-MNLI": None,  # auto-detect
        "roberta-large-mnli": None,
    }

    def __init__(self, model_name: str, temperature: float = 2.5,
                 device: str = "cpu") -> None:
        remap = self.KNOWN_MODELS.get(model_name)
        super().__init__(model_name, label_remap=remap,
                         temperature=temperature, device=device)


class QAModelWrapper:
    """QA model wrapper returning binary [p_wrong, p_correct] distributions.

    Uses span confidence (max start_prob * end_prob) as the correctness
    probability, enabling SUA computation with a 2-class output space.

    Args:
        model_name: HuggingFace QA model identifier.
        temperature: Temperature for softmax scaling.
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_name: str = "deepset/bert-base-uncased-squad2",
        temperature: float = 2.0,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.device = device
        self._tok = None
        self._mdl = None

    def load(self) -> "QAModelWrapper":
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._mdl = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name
        ).to(self.device)
        self._mdl.eval()
        return self

    def get_binary_probs(self, question: str, context: str) -> np.ndarray:
        """Get [p_wrong, p_correct] from span confidence."""
        inp = self._tok(question, context, return_tensors="pt",
                        truncation=True, max_length=384,
                        padding=True).to(self.device)
        with torch.no_grad():
            out = self._mdl(**inp)
        sl = out.start_logits[0] / self.temperature
        el = out.end_logits[0] / self.temperature
        sp = F.softmax(sl, dim=-1).cpu().numpy()
        ep = F.softmax(el, dim=-1).cpu().numpy()
        n = len(sp)
        best = 0.0
        for s in range(n):
            for e in range(s, min(s + 30, n)):
                v = float(sp[s] * ep[e])
                if v > best:
                    best = v
        pc = float(np.clip(best, 0.05, 0.95))
        return np.array([1 - pc, pc])

    def get_answer_text(self, question: str, context: str) -> str:
        """Get predicted answer span text."""
        inp = self._tok(question, context, return_tensors="pt",
                        truncation=True, max_length=384).to(self.device)
        with torch.no_grad():
            out = self._mdl(**inp)
        s = out.start_logits[0].argmax().item()
        e = out.end_logits[0].argmax().item()
        return self._tok.decode(
            inp["input_ids"][0][s:e + 1], skip_special_tokens=True
        ).strip()
