"""BM25 retrieval over a fixed evidence pool.

rank_bm25 does no preprocessing, so we lowercase, tokenize and drop stop words
ourselves. Each pool sentence is indexed together with its page title, because
many Wikipedia sentences start with a pronoun ("He was born in ...").
"""
import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .preprocess import normalize

STOP = set("""a an and are as at be been but by for from has have he her his i in is it
its of on or she that the their they this to was were which who will with""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


def title_of(doc_id: str) -> str:
    """'Jackie_-LRB-2016_film-RRB-' -> 'Jackie (2016 film)'."""
    return normalize(doc_id.replace("_", " ")).text


@dataclass
class Evidence:
    doc_id: str
    sentence_id: int
    text: str        # normalized sentence, prefixed with its page title
    score: float = 0.0

    @property
    def source_id(self) -> str:
        return f"{self.doc_id}:{self.sentence_id}"


def evidence_text(doc_id: str, sentence: str) -> str:
    title = title_of(doc_id)
    sent = normalize(sentence).text
    return sent if title.lower() in sent.lower() else f"{title}: {sent}"


class Index:
    def __init__(self, pool: list[dict]):
        self.items = [Evidence(p["doc_id"], int(p["sentence_id"]),
                               evidence_text(p["doc_id"], p["text"])) for p in pool]
        self.bm25 = BM25Okapi([tokenize(e.text) for e in self.items])

    def top_k(self, claim: str, k: int = 5) -> list[Evidence]:
        q = tokenize(normalize(claim).text)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [Evidence(self.items[i].doc_id, self.items[i].sentence_id,
                         self.items[i].text, float(scores[i])) for i in idx]

    @classmethod
    def from_file(cls, pool_path: Path, cache_dir: Path | None = None) -> "Index":
        """Build from a pool .jsonl, reusing a pickle keyed by the file's hash."""
        raw = pool_path.read_bytes()
        cache = None
        if cache_dir is not None:
            digest = hashlib.sha1(raw).hexdigest()[:12]
            cache = cache_dir / f"{pool_path.stem}-{digest}.pkl"
            if cache.exists():
                with open(cache, "rb") as f:
                    return pickle.load(f)
        index = cls([json.loads(line) for line in raw.decode().splitlines() if line.strip()])
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(index, f)
        return index
