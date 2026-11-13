"""Self-relabel agreement study (stories S0-3 and S1-4).

A person labels the same 100 FEVER-dev claims twice, days apart, without seeing
the FEVER label. Agreement is reported as Cohen's kappa with a bootstrap CI.

    python -m factcheck.relabel export            # data/relabel/pass1.csv
    python -m factcheck.relabel export --pass 2   # same claims, new order
    python -m factcheck.relabel score --pass1-date 2026-11-15 --pass2-date 2026-11-19

Fill the `label` column with SUPPORTS, REFUTES or NOT ENOUGH INFO (S, R, N or
NEI also accepted). Judge only from the evidence shown, not from world knowledge.
"""
import argparse
import csv
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from . import LABELS, config
from . import evaluate as ev

S, R, N = LABELS
SHEET_DIR = config.REPO / "data" / "relabel"
N_ITEMS = 100
SEED = 13
COLUMNS = ["item", "id", "claim", "evidence", "label", "notes"]
SHORTHAND = {"S": S, "R": R, "N": N, "NEI": N, "SUPPORTS": S, "REFUTES": R,
             "NOT ENOUGH INFO": N, "NOT_ENOUGH_INFO": N}
FEVER_KAPPA = 0.6841  # Fleiss kappa reported for FEVER (several annotators)


def parse_label(raw: str) -> str:
    key = raw.strip().upper()
    if key not in SHORTHAND:
        raise ValueError(f"unknown label {raw!r}; use SUPPORTS, REFUTES or NOT ENOUGH INFO")
    return SHORTHAND[key]


def build_sheet(rows, evidence_for, n: int = N_ITEMS, seed: int = SEED,
                order_seed: int | None = None) -> list[dict]:
    """Pick n claims stratified by label, return sheet rows with no gold label.

    `evidence_for(row) -> str` gives the sentence to show. `order_seed` reshuffles
    the same picks for a later pass."""
    rng = random.Random(seed)
    by_label = {lab: sorted((r for r in rows if r.label == lab and not r.is_opinion),
                            key=lambda r: r.id) for lab in LABELS}
    quota = {lab: n // len(LABELS) + (i < n % len(LABELS)) for i, lab in enumerate(LABELS)}
    picked = [r for lab in LABELS for r in rng.sample(by_label[lab], quota[lab])]
    random.Random(seed if order_seed is None else order_seed).shuffle(picked)
    return [{"item": i + 1, "id": r.id, "claim": r.claim, "evidence": evidence_for(r),
             "label": "", "notes": ""} for i, r in enumerate(picked)]


def write_sheet(sheet: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(sheet)


def read_labels(path: Path) -> dict[str, str]:
    """id -> label. Fails on blanks or unknown labels, listing the items."""
    labels, blank, bad = {}, [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row["label"].strip():
                blank.append(row["item"])
                continue
            try:
                labels[row["id"]] = parse_label(row["label"])
            except ValueError:
                bad.append(f"{row['item']}={row['label']!r}")
    if blank or bad:
        raise ValueError(f"{path.name}: blank items {blank[:10]}{'...' if len(blank) > 10 else ''}"
                         f"; invalid {bad[:10]}")
    return labels


def kappa_ci(a: list[str], b: list[str]) -> tuple[float, float, float]:
    items = list(zip(a, b, strict=True))
    return ev.bootstrap_ci(lambda it: ev.cohen_kappa([x for x, _ in it], [y for _, y in it]),
                           items)


def _evidence_fn():
    """Gold sentence for SUPPORTS/REFUTES; top BM25 sentence for NEI (no gold)."""
    from .retrieve import evidence_text
    from .run import _get_index

    def fn(r):
        if r.evidence:
            e = r.evidence[0]
            return evidence_text(e["doc_id"], e["text"])
        hits = _get_index().top_k(r.claim, 1)
        return hits[0].text if hits else ""
    return fn


def export(pass_no: int) -> Path:
    from .ingest import load
    d = config.setup_caches()
    rows = load(d / "processed" / "fever_dev.jsonl")
    sheet = build_sheet(rows, _evidence_fn(), order_seed=SEED + pass_no - 1)
    path = SHEET_DIR / f"pass{pass_no}.csv"
    if path.exists() and any(r.get("label", "").strip() for r in csv.DictReader(open(path))):
        sys.exit(f"{path} already has labels; refusing to overwrite it.")
    write_sheet(sheet, path)
    return path


def score(pass1_date: str | None, pass2_date: str | None, sheet_dir: Path = SHEET_DIR,
          out: Path | None = None) -> str:
    from .ingest import load
    d = config.setup_caches()
    gold = {r.id: r.label for r in load(d / "processed" / "fever_dev.jsonl")}
    p1 = read_labels(sheet_dir / "pass1.csv")
    p2 = read_labels(sheet_dir / "pass2.csv")
    if set(p1) != set(p2):
        raise ValueError("pass1 and pass2 cover different claims")
    ids = sorted(p1)
    a, b, g = [p1[i] for i in ids], [p2[i] for i in ids], [gold[i] for i in ids]
    n = len(ids)
    gap = ""
    if pass1_date and pass2_date:
        days = (date.fromisoformat(pass2_date) - date.fromisoformat(pass1_date)).days
        gap = f"{days} days (pass 1 on {pass1_date}, pass 2 on {pass2_date})"
    rows = [("Pass 1 vs pass 2 (intra-annotator)", kappa_ci(a, b), ev.accuracy(a, b)),
            ("Pass 1 vs FEVER label", kappa_ci(a, g), ev.accuracy(g, a)),
            ("Pass 2 vs FEVER label", kappa_ci(b, g), ev.accuracy(g, b))]
    L = ["# Label quality: self-relabel agreement", "",
         f"*Generated {date.today().isoformat()} by `python -m factcheck.relabel score`.*", "",
         f"- Items: **{n}** FEVER-dev claims, stratified by FEVER label, seed {SEED}. Opinion-"
         "flagged claims excluded.",
         "- Annotators: **1 person, labelling twice** (intra-annotator). FEVER labels are the "
         "released crowd labels.",
         f"- Gap between passes: {gap or 'not recorded'}.",
         "- Blind: the sheets show claim and evidence only; pass 2 is in a different order.",
         "- Bootstrap: 1,000 resamples of items, seed 13, percentile 95% CI.", "",
         "| Comparison | Cohen's κ | 95% CI | Raw agreement | n |", "|---|---|---|---|---|"]
    for name, (k, lo, hi), acc in rows:
        L.append(f"| {name} | {k:.3f} | [{lo:.3f}, {hi:.3f}] | {acc:.3f} | {n} |")
    L += ["", "Pass 1 (rows) vs pass 2 (columns), counts:", "",
          "| pass 1 \\ pass 2 | SUPPORTS | REFUTES | NEI |", "|---|---|---|---|"]
    for lab, row in zip(LABELS, ev.confusion(a, b), strict=True):
        L.append(f"| {lab} | " + " | ".join(map(str, row)) + " |")
    L += ["", f"Label mix: pass 1 {dict(Counter(a))}; pass 2 {dict(Counter(b))}.", "",
          "## How to read this",
          f"- Agreement of one person with themselves is easier to make high than agreement "
          f"between different people. Don't compare it directly with FEVER's Fleiss κ = "
          f"{FEVER_KAPPA} (several annotators), AVeriTeC's κ = 0.619, or SelfCheckGPT's 0.595 "
          "(3-class).",
          "- A short gap between passes inflates intra-annotator κ, because people remember "
          "their earlier answers.",
          "- Agreement with the FEVER label measures how closely this annotator follows FEVER's "
          "guidelines on these sentences. It is not an accuracy figure for either side.", ""]
    md = "\n".join(L)
    (out or config.REPO / "docs" / "label_quality.md").write_text(md)
    return md


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--pass", dest="pass_no", type=int, choices=[1, 2], default=1)
    sc = sub.add_parser("score")
    sc.add_argument("--pass1-date")
    sc.add_argument("--pass2-date")
    args = ap.parse_args(argv)
    if args.cmd == "export":
        print(f"wrote {export(args.pass_no)}")
    else:
        try:
            print(score(args.pass1_date, args.pass2_date))
        except (ValueError, FileNotFoundError) as e:
            sys.exit(str(e))


if __name__ == "__main__":
    main()
