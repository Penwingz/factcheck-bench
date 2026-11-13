"""Metrics, baselines and bootstrap confidence intervals.

Each metric is reported on its own with n and a percentile bootstrap 95% CI
(resampling claims). Retrieval, label and span quality are never blended.
"""
from collections import Counter

import numpy as np

from . import LABELS


def per_class_prf(gold: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def macro_f1(gold: list[str], pred: list[str]) -> float:
    """Mean F1 over the three labels (a label absent from gold and pred scores 0)."""
    return float(np.mean([v["f1"] for v in per_class_prf(gold, pred).values()]))


def accuracy(gold: list[str], pred: list[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0


def confusion(gold: list[str], pred: list[str]) -> list[list[int]]:
    """Rows = gold, columns = predicted, both in LABELS order."""
    idx = {lab: i for i, lab in enumerate(LABELS)}
    m = [[0] * len(LABELS) for _ in LABELS]
    for g, p in zip(gold, pred):
        m[idx[g]][idx[p]] += 1
    return m


def recall_hits(gold_sets: list[set], retrieved: list[list], k: int) -> list[int]:
    """1 per claim if any gold sentence id is in the top k, else 0."""
    return [int(bool(g & set(r[:k]))) for g, r in zip(gold_sets, retrieved)]


def fever_hits(gold: list[str], pred: list[str], gold_sets: list[set],
               retrieved: list[list], k: int = 5) -> list[int]:
    """FEVER-style strict score per claim: label correct, and for SUPPORTS/REFUTES at
    least one gold sentence in the top k. Single-sentence approximation: the official
    score needs a complete gold evidence set, which matters for multi-sentence sets."""
    return [int(g == p and (g == LABELS[2] or bool(gs & set(r[:k]))))
            for g, p, gs, r in zip(gold, pred, gold_sets, retrieved)]


def span_char_counts(gold: list[list[int]], pred: list[list[int]]) -> tuple[int, int, int]:
    """(overlap, predicted chars, gold chars) for one claim; spans are [start, end)."""
    g = set().union(*[range(s, e) for s, e in gold]) if gold else set()
    p = set().union(*[range(s, e) for s, e in pred]) if pred else set()
    return len(g & p), len(p), len(g)


def span_f1_from_counts(counts: list[tuple[int, int, int]]) -> dict[str, float]:
    """Micro character-level P/R/F1 summed over claims."""
    inter, npred, ngold = (sum(c[i] for c in counts) for i in range(3))
    prec = inter / npred if npred else 0.0
    rec = inter / ngold if ngold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def span_char_f1(gold: list[list[list[int]]], pred: list[list[list[int]]]) -> dict[str, float]:
    return span_f1_from_counts([span_char_counts(g, p) for g, p in zip(gold, pred)])


def bootstrap_ci(fn, items: list, n_boot: int = 1000, seed: int = 13,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile CI of fn(items) by resampling items."""
    point = fn(items)
    if not items:
        return point, point, point
    rng = np.random.default_rng(seed)
    n = len(items)
    stats = [fn([items[i] for i in rng.integers(0, n, n)]) for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def mean(xs: list) -> float:
    return float(np.mean(xs)) if xs else 0.0


# Baselines ---------------------------------------------------------------

def majority_baseline(train_labels: list[str], n: int) -> list[str]:
    top = Counter(train_labels).most_common(1)[0][0]
    return [top] * n


def claim_only_baseline(train_claims, train_labels, test_claims, seed: int = 13) -> list[str]:
    """TF-IDF + logistic regression on the claim text alone (no evidence). If this
    beats the majority class, the labels leak through claim wording."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(vec.fit_transform(train_claims), train_labels)
    return list(clf.predict(vec.transform(test_claims)))
