"""NLI verdict: cross-encoder/nli-deberta-v3-xsmall, inference only, on CPU.

The label order is read from model.config.id2label, never hardcoded.
contradiction -> REFUTES, entailment -> SUPPORTS, neutral -> NOT ENOUGH INFO.
Probabilities are softmax outputs and are UNCALIBRATED; they are not a
confidence score.
"""
from dataclasses import dataclass

import numpy as np

from . import LABELS, config

S, R, N = LABELS
NLI_TO_LABEL = {"contradiction": R, "entailment": S, "neutral": N}


def label_order(model_config) -> list[str]:
    """Our label for each model output index, from config.id2label."""
    id2label = {int(k): v for k, v in model_config.id2label.items()}
    order = []
    for i in range(len(id2label)):
        name = id2label[i].lower()
        match = [ours for nli, ours in NLI_TO_LABEL.items() if name.startswith(nli[:6])]
        if len(match) != 1:
            raise ValueError(f"unexpected NLI label {id2label[i]!r}")
        order.append(match[0])
    if sorted(order) != sorted(LABELS):
        raise ValueError(f"id2label does not cover all three labels: {id2label}")
    return order


@dataclass
class Thresholds:
    supports: float = 0.5   # min P(entailment) on the best sentence to say SUPPORTS
    refutes: float = 0.5    # min P(contradiction) to say REFUTES


class NLI:
    def __init__(self, model_name: str | None = None, batch_size: int = 32):
        config.setup_caches()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        name = model_name or config.load()["verdict"]["model"]
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        order = label_order(self.model.config)
        # column permutation from model order to LABELS order
        self.perm = [order.index(lab) for lab in LABELS]
        self.batch_size = batch_size

    def probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """(premise, hypothesis) pairs -> array (n, 3) in LABELS order."""
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                enc = self.tokenizer([p for p, _ in batch], [h for _, h in batch],
                                     padding=True, truncation=True, max_length=256,
                                     return_tensors="pt")
                logits = self.model(**enc).logits
                out.append(self.torch.softmax(logits, dim=-1).numpy()[:, self.perm])
        return np.concatenate(out) if out else np.zeros((0, 3))


def aggregate(probs: np.ndarray, th: Thresholds) -> tuple[str, int]:
    """Combine per-sentence NLI rows (n, 3) into one label.

    Returns (label, index of the deciding sentence, or -1 if none)."""
    if len(probs) == 0:
        return N, -1
    e, c = probs[:, 0], probs[:, 1]
    ie, ic = int(e.argmax()), int(c.argmax())
    sup, ref = e[ie] >= th.supports, c[ic] >= th.refutes
    if sup and (not ref or e[ie] >= c[ic]):
        return S, ie
    if ref:
        return R, ic
    return N, int(np.maximum(e, c).argmax())


def tune(prob_sets: list[np.ndarray], gold: list[str], grid=None) -> tuple[Thresholds, float]:
    """Grid-search thresholds for macro-F1. Call this on the dev split only."""
    from .evaluate import macro_f1
    if grid is None:
        grid = np.concatenate([np.round(np.arange(0.3, 0.96, 0.05), 2), [0.97, 0.98, 0.99, 0.995]])
    best = (Thresholds(), -1.0)
    for ts in grid:
        for tr in grid:
            th = Thresholds(float(ts), float(tr))
            f1 = macro_f1(gold, [aggregate(p, th)[0] for p in prob_sets])
            if f1 > best[1] + 1e-9:
                best = (th, f1)
    return best
