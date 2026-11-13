"""Download source datasets, map them to schema v1 rows and validate them.

Sources:
- FEVER claims with gold evidence text: copenlu/fever_gold_evidence (valid.jsonl).
  CC BY-SA 3.0 / GPL-3.0 per the dataset card. Avoids the 7.25 GB wiki dump.
- Climate-FEVER: tdiggelm/climate_fever (licence listed as unknown; eval only).

Usage: python -m factcheck.ingest
"""
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import LABELS, SCHEMA_VERSION, config

FEVER_SRC = ("copenlu/fever_gold_evidence", "valid.jsonl")
FEVER_DISTRACTOR_SRC = ("copenlu/fever_gold_evidence", "train.jsonl")  # 97 MB
N_DISTRACTORS = 24000
CLIMATE_SRC = ("tdiggelm/climate_fever", "data/test-00000-of-00001.parquet")
CLIMATE_LABELS = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}  # 3 = DISPUTED, dropped
FEVER_SAMPLE = 3000
SEED = 13


@dataclass
class Row:
    id: str
    source_id: str
    raw_text: str
    claim: str
    claim_start: int
    claim_end: int
    label: str
    evidence: list = field(default_factory=list)  # [{doc_id, sentence_id, text}]
    wrong_spans: list = field(default_factory=list)  # [{start, end}] offsets into claim
    is_opinion: bool = False
    annotator: str = ""
    annotated_at: str = ""
    schema_version: str = SCHEMA_VERSION


class ValidationError(ValueError):
    pass


def validate(row: Row, require_refutes_span: bool = False) -> None:
    """Schema v1 checks. FEVER has no gold spans, so the REFUTES-needs-a-span
    rule only applies to hand-labelled span data (require_refutes_span=True)."""
    if row.label not in LABELS:
        raise ValidationError(f"{row.id}: bad label {row.label!r}")
    if row.raw_text[row.claim_start:row.claim_end] != row.claim:
        raise ValidationError(f"{row.id}: claim offsets do not round-trip")
    for s in row.wrong_spans:
        if not (0 <= s["start"] < s["end"] <= len(row.claim)):
            raise ValidationError(f"{row.id}: wrong span {s} outside claim")
    if row.wrong_spans and row.label != "REFUTES":
        raise ValidationError(f"{row.id}: wrong spans on a non-REFUTES row")
    if require_refutes_span and row.label == "REFUTES" and not row.wrong_spans:
        raise ValidationError(f"{row.id}: REFUTES row without a wrong span")
    if row.schema_version != SCHEMA_VERSION:
        raise ValidationError(f"{row.id}: schema version {row.schema_version}")


def load(path: Path, require_refutes_span: bool = False) -> list[Row]:
    rows = []
    with open(path) as f:
        for line in f:
            row = Row(**json.loads(line))
            validate(row, require_refutes_span)
            rows.append(row)
    return rows


def save(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def _download(src: tuple[str, str], raw_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    repo, filename = src
    return Path(hf_hub_download(repo, filename, repo_type="dataset",
                                local_dir=raw_dir / repo.replace("/", "__")))


def fever_rows(raw_path: Path, n: int = FEVER_SAMPLE, seed: int = SEED):
    """Return (sampled schema rows, pool sentences from all FEVER rows).

    The pool holds gold sentences of every row in the file plus the retrieved
    distractor sentences attached to NEI rows, so most pool sentences belong to
    claims outside our sample and act as distractors."""
    by_label = defaultdict(list)
    pool = {}
    with open(raw_path) as f:
        for line in f:
            r = json.loads(line)
            by_label[r["label"]].append(r)
            for page, sid, text in r["evidence"]:
                pool[f"{page}:{sid}"] = {"doc_id": page, "sentence_id": int(sid), "text": text}
    rng = random.Random(seed)
    per_label = n // len(LABELS)
    sampled = []
    for label in LABELS:
        items = sorted(by_label[label], key=lambda r: r["id"])
        sampled += rng.sample(items, per_label)
    rows = []
    for r in sampled:
        ev = [{"doc_id": p, "sentence_id": int(s), "text": t} for p, s, t in r["evidence"]]
        # NEI evidence in this file is retrieved context, not gold: keep it out of
        # the row but use its page as the source so related claims stay together.
        rows.append(Row(
            id=f"fever-{r['id']}",
            source_id=ev[0]["doc_id"] if ev else f"nosource-{r['id']}",
            raw_text=r["claim"], claim=r["claim"], claim_start=0, claim_end=len(r["claim"]),
            label=r["label"],
            evidence=ev if r["label"] != "NOT ENOUGH INFO" else [],
            annotator="fever-crowd", annotated_at="2018 (FEVER release)",
        ))
    return rows, list(pool.values())


def add_distractors(pool: list[dict], raw_path: Path, n: int = N_DISTRACTORS,
                    seed: int = SEED) -> list[dict]:
    """Add n sentences from other FEVER claims so retrieval has real competition."""
    seen = {(p["doc_id"], p["sentence_id"]) for p in pool}
    extra = {}
    with open(raw_path) as f:
        for line in f:
            for page, sid, text in json.loads(line)["evidence"]:
                if (page, int(sid)) not in seen:
                    extra[(page, int(sid))] = text
    keys = sorted(extra)
    picked = random.Random(seed).sample(keys, min(n, len(keys)))
    return pool + [{"doc_id": p, "sentence_id": s, "text": extra[(p, s)]} for p, s in picked]


def climate_rows(raw_path: Path):
    import pyarrow.parquet as pq
    rows, pool, dropped = [], {}, Counter()
    for r in pq.read_table(raw_path).to_pylist():
        if r["claim_label"] not in CLIMATE_LABELS:
            dropped["DISPUTED"] += 1
            continue
        label = CLIMATE_LABELS[r["claim_label"]]
        gold = []
        for e in r["evidences"]:
            doc, _, sid = e["evidence_id"].rpartition(":")
            item = {"doc_id": doc, "sentence_id": int(sid), "text": e["evidence"]}
            pool[e["evidence_id"]] = item
            if label != "NOT ENOUGH INFO" and CLIMATE_LABELS.get(e["evidence_label"]) == label:
                gold.append(item)
        rows.append(Row(
            id=f"climate-{r['claim_id']}", source_id=f"climate-{r['claim_id']}",
            raw_text=r["claim"], claim=r["claim"], claim_start=0, claim_end=len(r["claim"]),
            label=label, evidence=gold,
            annotator="climate-fever-crowd (5 votes per evidence)", annotated_at="2020 (release)",
        ))
    return rows, list(pool.values()), dropped


def main() -> None:
    d = config.setup_caches()
    fever, fpool = fever_rows(_download(FEVER_SRC, d / "raw"))
    fpool = add_distractors(fpool, _download(FEVER_DISTRACTOR_SRC, d / "raw"))
    climate, cpool, dropped = climate_rows(_download(CLIMATE_SRC, d / "raw"))
    for r in fever + climate:
        validate(r)
    save(fever, d / "processed" / "fever_rows.jsonl")
    save(climate, d / "processed" / "climate_rows.jsonl")
    for name, pool in (("fever_pool", fpool), ("climate_pool", cpool)):
        with open(d / "processed" / f"{name}.jsonl", "w") as f:
            for p in pool:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("fever", len(fever), Counter(r.label for r in fever), "pool", len(fpool))
    print("climate", len(climate), Counter(r.label for r in climate), "pool", len(cpool),
          "dropped", dict(dropped))


if __name__ == "__main__":
    main()
