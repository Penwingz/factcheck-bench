"""Integration function check() and the evaluation CLI.

    python -m factcheck.run --split dev                     # tune thresholds, report
    python -m factcheck.run --split test                    # uses dev thresholds
    python -m factcheck.run --split test --dataset climate  # out-of-domain, eval only
"""
import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
from dataclasses import asdict
from datetime import date

from . import LABELS, SCHEMA_VERSION, config, evaluate as ev, spans
from .ingest import load
from .preprocess import is_opinion, normalize, split_claims
from .retrieve import Evidence, Index
from .verdict import NLI, Thresholds, aggregate, tune

S, R, N = LABELS
RESULTS = config.REPO / "results"
TOP_K = 5
MIN_PAIRS_PER_SEC = 5.0
CAP_IF_SLOW = 300

_nli = None
_index = None


def _get_nli() -> NLI:
    global _nli
    if _nli is None:
        _nli = NLI()
    return _nli


def _get_index() -> Index:
    global _index
    if _index is None:
        d = config.setup_caches()
        _index = Index.from_file(d / "processed" / "fever_pool.jsonl", d / "index")
    return _index


def load_thresholds() -> Thresholds:
    p = RESULTS / "thresholds.json"
    if p.exists():
        d = json.loads(p.read_text())
        return Thresholds(d["supports"], d["refutes"])
    return Thresholds()


def _result(label, scores, evidence_used, wrong_spans, opinion) -> dict:
    return {"label": label, "scores": scores, "evidence_used": evidence_used,
            "wrong_spans": wrong_spans, "is_opinion": opinion,
            "schema_version": SCHEMA_VERSION}


def check(claim: str, evidence: list[str] | None = None,
          thresholds: Thresholds | None = None) -> dict:
    """Check one claim.

    If `evidence` is given, retrieval is skipped and the claim is checked against
    those passages. Otherwise the top-5 BM25 sentences from the fixed pool are used.

    Returns label, scores (raw NLI class probabilities for the deciding evidence
    sentence; UNCALIBRATED, not a confidence score), evidence_used (text, source_id,
    per-sentence scores, which sentence decided), wrong_spans ([start, end]
    character offsets into `claim`), is_opinion, schema_version.

    The label means "supported or refuted by the evidence used", not "true".
    """
    if is_opinion(claim):
        return _result(N, None, [], [], True)
    if evidence is not None:
        items = [Evidence("provided", i, normalize(t).text) for i, t in enumerate(evidence)]
    else:
        items = _get_index().top_k(claim, TOP_K)
    probs = _get_nli().probs([(e.text, claim) for e in items])
    label, best = aggregate(probs, thresholds or load_thresholds())
    used = [{"text": e.text, "source_id": e.source_id,
             "scores": dict(zip(LABELS, map(float, p))), "decisive": i == best}
            for i, (e, p) in enumerate(zip(items, probs))]
    scores = dict(zip(LABELS, map(float, probs[best]))) if best >= 0 else None
    wrong = spans.find(claim, items[best].text) if label == R else []
    return _result(label, scores, used, wrong, False)


def check_text(text: str, evidence: list[str] | None = None) -> list[dict]:
    """Split free text into claims and check each; adds claim_start/claim_end."""
    out = []
    for s, e in split_claims(text):
        r = check(text[s:e], evidence)
        r.update(claim=text[s:e], claim_start=s, claim_end=e)
        out.append(r)
    return out


# Evaluation ------------------------------------------------------------------

class ProbCache:
    """NLI probabilities keyed by (premise, hypothesis) hash, stored on the data dir."""

    def __init__(self, path):
        self.path = path
        self.data = pickle.loads(path.read_bytes()) if path.exists() else {}

    @staticmethod
    def key(p, h):
        return hashlib.sha1(f"{p}\x00{h}".encode()).hexdigest()

    def fill(self, nli: NLI, pairs):
        todo = list({self.key(p, h): (p, h) for p, h in pairs
                     if self.key(p, h) not in self.data}.items())
        t = time.perf_counter()
        if todo:
            for (k, _), row in zip(todo, nli.probs([ph for _, ph in todo])):
                self.data[k] = row
            self.path.write_bytes(pickle.dumps(self.data))
        return len(todo), time.perf_counter() - t

    def get(self, p, h):
        return self.data[self.key(p, h)]


def measure_throughput(nli: NLI, pairs, n: int = 96) -> float:
    sample = pairs[:n]
    nli.probs(sample[:8])  # warm-up
    t = time.perf_counter()
    nli.probs(sample)
    return len(sample) / (time.perf_counter() - t)


def ci_line(name, point_lo_hi, n, split, annotators, fmt="{:.3f}") -> str:
    p, lo, hi = point_lo_hi
    return (f"| {name} | {fmt.format(p)} | [{fmt.format(lo)}, {fmt.format(hi)}] | {n} | "
            f"{split} | {annotators} |")


def label_metrics(gold, pred):
    items = list(zip(gold, pred))
    return {
        "macro_f1": ev.bootstrap_ci(lambda it: ev.macro_f1([g for g, _ in it], [p for _, p in it]), items),
        "accuracy": ev.bootstrap_ci(lambda it: ev.accuracy([g for g, _ in it], [p for _, p in it]), items),
        "per_class": ev.per_class_prf(gold, pred),
        "confusion": ev.confusion(gold, pred),
        "n": len(gold),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--dataset", choices=["fever", "climate"], default="fever")
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N claims")
    ap.add_argument("--full-wiki", action="store_true",
                    help="use the full FEVER Wikipedia dump (~1.7 GB zipped, 7.25 GB "
                         "expanded). Not implemented in this prototype; off by default.")
    args = ap.parse_args(argv)
    if args.full_wiki:
        sys.exit("--full-wiki is not implemented: it needs a >500 MB download, which "
                 "must be approved first. The default small pool is used otherwise.")
    if args.dataset == "climate" and args.split == "dev":
        sys.exit("Climate-FEVER is evaluation only: use --split test (thresholds come from FEVER dev).")

    d = config.setup_caches()
    proc = d / "processed"
    if args.dataset == "fever":
        rows = load(proc / f"fever_{args.split}.jsonl")
        pool = proc / "fever_pool.jsonl"
        annotators = "FEVER crowd labels, 1 label per claim as released"
    else:
        rows = load(proc / "climate_test.jsonl")
        pool = proc / "climate_pool.jsonl"
        annotators = "Climate-FEVER crowd, 5 votes per evidence sentence"
    split_name = f"{args.dataset}-{args.split}"
    train = load(proc / "fever_train.jsonl")
    index = Index.from_file(pool, d / "index")
    nli = _get_nli()

    # Retrieval for every claim (opinions included, so recall@k covers all S/R rows)
    t0 = time.perf_counter()
    retrieved = [index.top_k(r.claim, TOP_K) for r in rows]
    retrieval_qps = len(rows) / (time.perf_counter() - t0)

    pairs = [(e.text, r.claim) for r, hits in zip(rows, retrieved) for e in hits]
    pps = measure_throughput(nli, pairs)
    capped = False
    if args.limit or pps < MIN_PAIRS_PER_SEC:
        cap = args.limit or CAP_IF_SLOW
        capped = len(rows) > cap
        rows, retrieved = rows[:cap], retrieved[:cap]
        pairs = [(e.text, r.claim) for r, hits in zip(rows, retrieved) for e in hits]

    cache = ProbCache(d / "index" / "nli_cache.pkl")
    n_new, secs = cache.fill(nli, pairs)
    prob_sets = []
    import numpy as np
    for r, hits in zip(rows, retrieved):
        prob_sets.append(np.array([cache.get(e.text, r.claim) for e in hits]).reshape(-1, 3))

    gold = [r.label for r in rows]
    opinion = [r.is_opinion for r in rows]

    def predict(th):
        return [N if op else aggregate(p, th)[0] for p, op in zip(prob_sets, opinion)]

    default_th = Thresholds()
    if args.split == "dev" and args.dataset == "fever":
        tuned_th, _ = tune([p for p, op in zip(prob_sets, opinion) if not op],
                           [g for g, op in zip(gold, opinion) if not op])
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "thresholds.json").write_text(json.dumps(
            {**asdict(tuned_th), "tuned_on": "fever-dev", "metric": "macro-F1",
             "date": date.today().isoformat()}, indent=2) + "\n")
    else:
        if not (RESULTS / "thresholds.json").exists():
            sys.exit("No results/thresholds.json: run --split dev first.")
        tuned_th = load_thresholds()

    pred_default, pred_tuned = predict(default_th), predict(tuned_th)
    pred_major = ev.majority_baseline([r.label for r in train], len(rows))
    pred_claim = ev.claim_only_baseline([r.claim for r in train], [r.label for r in train],
                                        [r.claim for r in rows])
    systems = {
        "NLI system, thresholds 0.5/0.5 (untuned)": pred_default,
        f"NLI system, thresholds {tuned_th.supports:.2f}/{tuned_th.refutes:.2f} (tuned on FEVER dev)": pred_tuned,
        "Majority class (from FEVER train)": pred_major,
        "Claim-only TF-IDF + LR (trained on FEVER train)": pred_claim,
    }
    lm = {name: label_metrics(gold, p) for name, p in systems.items()}

    # Retrieval, scored only on SUPPORTS/REFUTES claims that have gold sentences
    gold_sets = [{(e["doc_id"], e["sentence_id"]) for e in r.evidence} for r in rows]
    ret_ids = [[(e.doc_id, e.sentence_id) for e in hits] for hits in retrieved]
    has_gold = [i for i, r in enumerate(rows) if r.label != N and gold_sets[i]]
    recall = {k: ev.bootstrap_ci(ev.mean, ev.recall_hits([gold_sets[i] for i in has_gold],
                                                         [ret_ids[i] for i in has_gold], k))
              for k in (1, 3, 5)}
    fever = ev.bootstrap_ci(ev.mean, ev.fever_hits(gold, pred_tuned, gold_sets, ret_ids, TOP_K))

    # Spans on the hand-built set: rule alone (gold evidence), and end-to-end check()
    hand = load(config.REPO / "data" / "span_gold_handbuilt.jsonl", require_refutes_span=True)
    hgold = [[[s["start"], s["end"]] for s in r.wrong_spans] for r in hand]
    rule_pred = [spans.find(r.claim, r.evidence[0]["text"]) for r in hand]
    e2e = [check(r.claim, [r.evidence[0]["text"]], tuned_th) for r in hand]
    e2e_pred = [x["wrong_spans"] for x in e2e]
    span_rule = ev.bootstrap_ci(lambda c: ev.span_f1_from_counts(c)["f1"],
                                [ev.span_char_counts(g, p) for g, p in zip(hgold, rule_pred)])
    span_e2e = ev.bootstrap_ci(lambda c: ev.span_f1_from_counts(c)["f1"],
                               [ev.span_char_counts(g, p) for g, p in zip(hgold, e2e_pred)])
    span_rule_prf = ev.span_char_f1(hgold, rule_pred)
    span_e2e_prf = ev.span_char_f1(hgold, e2e_pred)
    hand_label_acc = ev.accuracy([r.label for r in hand], [x["label"] for x in e2e])

    metrics = {
        "split": split_name, "n_claims": len(rows), "capped": capped,
        "annotators": annotators, "date": date.today().isoformat(),
        "throughput": {"nli_pairs_per_sec": pps, "retrieval_queries_per_sec": retrieval_qps,
                       "new_pairs_scored": n_new, "seconds": secs,
                       "machine": f"{platform.machine()} {platform.system()}"},
        "thresholds": asdict(tuned_th),
        "label": {k: {**v, "macro_f1": list(v["macro_f1"]), "accuracy": list(v["accuracy"])}
                  for k, v in lm.items()},
        "retrieval": {"n": len(has_gold), **{f"recall@{k}": list(v) for k, v in recall.items()}},
        "fever_style_strict": {"n": len(rows), "score": list(fever)},
        "spans_handbuilt": {"n": len(hand), "n_refutes": sum(r.label == R for r in hand),
                            "rule_with_gold_evidence": {**span_rule_prf, "f1_ci": list(span_rule)},
                            "end_to_end_check": {**span_e2e_prf, "f1_ci": list(span_e2e),
                                                 "label_accuracy": hand_label_acc}},
        "opinion_flagged": sum(opinion),
    }
    RESULTS.mkdir(exist_ok=True)
    stem = "results" if split_name == "fever-dev" else f"results-{split_name}"
    (RESULTS / f"{stem.replace('results', 'metrics')}.json").write_text(json.dumps(metrics, indent=2))
    md = render(metrics, lm, rows, retrieved, prob_sets, pred_tuned, hand, e2e, tuned_th)
    (RESULTS / f"{stem}.md").write_text(md)
    print(md)


def render(m, lm, rows, retrieved, prob_sets, pred, hand, e2e, th) -> str:
    split, ann, n = m["split"], m["annotators"], m["n_claims"]
    tp = m["throughput"]
    hdr = ["| Metric | Value | 95% CI | n | Split | Annotators |", "|---|---|---|---|---|---|"]
    L = [f"# Results: {split}", "",
         f"*Generated {m['date']} by `python -m factcheck.run`. Bootstrap: 1,000 resamples of "
         "claims, seed 13, percentile 95% CI. Each number describes one system on one split; "
         "none of them is comparable with FEVER or AVeriTeC leaderboards.*", "",
         "## Run facts",
         f"- Split: **{split}**, claims evaluated: **{n}**"
         + (f" (capped at {CAP_IF_SLOW} because throughput was under {MIN_PAIRS_PER_SEC} pairs/s)"
            if m["capped"] else " (no cap; throughput was above 5 pairs/s)"),
         f"- Labels: {ann}. No re-annotation by this project.",
         f"- NLI throughput: **{tp['nli_pairs_per_sec']:.1f} claim-evidence pairs/s** on CPU "
         f"({tp['machine']}, batch 32, cross-encoder/nli-deberta-v3-xsmall). Retrieval: "
         f"{tp['retrieval_queries_per_sec']:.0f} queries/s.",
         f"- Thresholds used for the tuned system: SUPPORTS if max P(entailment) ≥ "
         f"{th.supports:.2f}, REFUTES if max P(contradiction) ≥ {th.refutes:.2f}, else NOT "
         "ENOUGH INFO. Tuned for macro-F1 on **FEVER dev only**.",
         f"- Claims flagged as opinion (predicted NOT ENOUGH INFO without checking): "
         f"{m['opinion_flagged']}", "",
         "## 1. Label quality (macro-F1 over 3 classes)"] + hdr
    for name, v in lm.items():
        L.append(ci_line(f"{name}: macro-F1", v["macro_f1"], n, split, ann))
        L.append(ci_line(f"{name}: accuracy", v["accuracy"], n, split, ann))
    for name, v in lm.items():
        L += ["", f"### {name}", "", "| Class | Precision | Recall | F1 | Support |",
              "|---|---|---|---|---|"]
        for lab, pc in v["per_class"].items():
            L.append(f"| {lab} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | "
                     f"{pc['support']} |")
        L += ["", "Confusion matrix (rows = gold, columns = predicted):", "",
              "| gold \\ pred | SUPPORTS | REFUTES | NEI |", "|---|---|---|---|"]
        for lab, row in zip(LABELS, v["confusion"]):
            L.append(f"| {lab} | " + " | ".join(map(str, row)) + " |")
    r = m["retrieval"]
    L += ["", "## 2. Retrieval (BM25, scored separately from the label)",
          "Hit rate: share of SUPPORTS/REFUTES claims with at least one gold evidence sentence "
          "in the top k. NOT ENOUGH INFO claims have no gold evidence and are excluded.", ""] + hdr
    for k in (1, 3, 5):
        L.append(ci_line(f"recall@{k}", r[f"recall@{k}"], r["n"], split, ann))
    L += ["", "## 3. FEVER-style strict score",
          "Label correct and, for SUPPORTS/REFUTES, a gold sentence in the top 5. This is a "
          "single-sentence approximation of the official FEVER score (which needs a complete "
          "evidence set), so it is not comparable with published FEVER scores.", ""] + hdr
    L.append(ci_line("FEVER-style strict score (tuned system)", m["fever_style_strict"]["score"],
                     n, split, ann))
    s = m["spans_handbuilt"]
    hann = "1 (Claude, hand-built; not a human annotator)"
    L += ["", "## 4. Wrong-span character F1 (hand-built set)",
          f"The public datasets have no span labels, so spans are scored only on "
          f"`data/span_gold_handbuilt.jsonl`: {s['n']} claims ({s['n_refutes']} REFUTES, the "
          "rest SUPPORTS). Three REFUTES cases are negation or common-noun swaps, which the "
          "rules are known not to handle. Treat this as a smoke test, not a benchmark.", ""] + hdr
    L.append(ci_line("Span char F1, rule only, given gold evidence", s["rule_with_gold_evidence"]["f1_ci"],
                     s["n"], "handbuilt", hann))
    L.append(ci_line("Span char F1, end-to-end check() with gold evidence",
                     s["end_to_end_check"]["f1_ci"], s["n"], "handbuilt", hann))
    L += ["", f"- Rule only: precision {s['rule_with_gold_evidence']['precision']:.3f}, recall "
          f"{s['rule_with_gold_evidence']['recall']:.3f}.",
          f"- End to end: precision {s['end_to_end_check']['precision']:.3f}, recall "
          f"{s['end_to_end_check']['recall']:.3f}; spans appear only when the label is "
          f"REFUTES. Label accuracy on these {s['n']} claims: "
          f"{s['end_to_end_check']['label_accuracy']:.3f}.", "",
          "| Claim | Gold label | Predicted | Gold span | Predicted spans |", "|---|---|---|---|---|"]
    for r_, x in zip(hand, e2e):
        g = ", ".join(r_.claim[a["start"]:a["end"]] for a in r_.wrong_spans) or "-"
        p = ", ".join(r_.claim[a:b] for a, b in x["wrong_spans"]) or "-"
        L.append(f"| {r_.claim} | {r_.label} | {x['label']} | {g} | {p} |")
    L += ["", "## 5. Worked examples (tuned system)", ""]
    shown, used_sources = 0, set()
    for want_correct in (True, False):
        for row, hits, probs, p in zip(rows, retrieved, prob_sets, pred):
            if ((p == row.label) != want_correct or row.is_opinion or not hits
                    or row.source_id in used_sources):
                continue
            used_sources.add(row.source_id)
            res_label, best = aggregate(probs, th)
            e = hits[best] if best >= 0 else hits[0]
            ok = "correct" if want_correct else "wrong"
            L += [f"- **{ok}**: `{row.claim}` (gold {row.label}, predicted {p})",
                  f"  - deciding evidence: `{e.text}` ({e.source_id})"]
            if p == R:
                L.append("  - wrong spans: " + (", ".join(
                    f"`{row.claim[a:b]}`" for a, b in spans.find(row.claim, e.text)) or "none"))
            shown += 1
            if shown % 3 == 0:
                break
    L += ["", "## Caveats",
          "- **Tuned on this split (dev only):** the tuned thresholds were chosen on FEVER dev, "
          "so FEVER-dev numbers for the tuned system are optimistic. The untuned 0.5/0.5 row "
          "is the fairer dev number; FEVER test and Climate-FEVER use the dev thresholds "
          "unchanged.",
          "- **Gold-seeded pool:** the FEVER pool (30,221 sentences) contains every gold sentence "
          "plus distractors, so retrieval is far easier than over all of Wikipedia. NOT ENOUGH "
          "INFO means \"not in our pool\".",
          "- **Balanced sample:** the 1,000-per-label sample does not match FEVER's real label "
          "mix.",
          "- **Labels:** one released crowd label per claim; this project did no re-annotation "
          "and has no agreement study (S0-3 not done).",
          "- **Claim-only baseline:** where it beats the majority class, claim wording alone "
          "predicts the label (a FEVER artefact). Compare the NLI system with it, not only with "
          "the majority class.",
          "- **Scores are uncalibrated** softmax outputs, not confidence scores.",
          "- **Spans:** rule-based, scored on a small Claude-built set. Negation, verb swaps and "
          "common-noun swaps are not handled.",
          "- **Climate-FEVER** licence is listed as unknown; it is used for evaluation only. "
          "DISPUTED claims were dropped.",
          "- **No leakage from training:** the NLI model was trained on SNLI and MNLI, not FEVER. "
          "Nothing was trained or fine-tuned here; thresholds are the only fitted values, plus "
          "the claim-only baseline trained on FEVER train.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
