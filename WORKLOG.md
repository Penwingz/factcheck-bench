# Work log

## 2026-11-14
**Changed**
- Set up the repo: `.gitignore` (reference folder, local config, data, venv), `config.example.toml`, `CLAUDE.md`, and a repo-local git identity.
- S0-1: ingest of FEVER (3,000-claim balanced sample) and Climate-FEVER into schema v1, with validation and the data card.
- S0-2: offset-preserving normalisation, claim splitter, opinion filter, de-dup, split by source. Writes `docs/preprocessing-log.md`. Splits: train 1,853 / dev 397 / test 396, with no source overlap.
- S1-2: BM25 retrieval over a 30,221-sentence pool.
- S1-3: rule-based wrong spans and a 20-claim hand-built span set.
- S1-1: NLI verdict, `check()` / `check_text()` integration API.
- S0-4 / S1-5: metrics, baselines, bootstrap CIs and the `python -m factcheck.run` CLI; results for FEVER dev, FEVER test and Climate-FEVER.

**Measured** (details in `results/`)
- NLI throughput: 76–110 pairs/s on CPU, so no dev cap was needed.
- FEVER dev, n=397, thresholds tuned there: macro-F1 0.635 [0.590, 0.684]. Untuned: 0.514 [0.469, 0.556].
- FEVER test, n=396: macro-F1 0.647 [0.601, 0.690], against claim-only 0.431 [0.381, 0.475] and majority 0.162. Recall@5 0.836 (n=250).
- Climate-FEVER, n=1,351: macro-F1 0.324 [0.299, 0.351], no better than claim-only (0.332).
- Spans (hand-built, n=20, by Claude): rule-only char F1 0.856 [0.640, 1.000]; end-to-end 0.647 [0.325, 0.857].

**Problems found**
- First de-dup pass dropped FEVER mutation pairs; fixed with a label check and a 0.9 threshold.
- First threshold search hit the edge of the grid; extended the grid.
- Committed the `.venv` symlink by mistake (it held the absolute SSD path); untracked it in the next commit.

**Next**
- NEI claims predicted REFUTES are the main error. Try a relevance gate (minimum BM25 score or entity overlap) before trusting a contradiction; tune it on dev.
- A human-labelled span set, and the self-relabel agreement study (S0-3).
- Slides with a limits slide (S0-5).

## 2026-11-14 (second session)
**Review**
- Bare `pytest`, the command in the brief, failed at collection because `factcheck` wasn't on `sys.path`. Added `pyproject.toml` with pytest `pythonpath`; it now passes.
- Rule 3 gap: per-class P/R/F1 and span P/R had no CIs. Added `evaluate.bootstrap_many`; every metric in `results/` now has a 95% CI.
- Lint (ruff E, F, I, B): fixed import order and long lines; every `zip()` is now `strict=True`. No unused imports or undefined names were found.
- Results regenerated. Thresholds unchanged (0.95/0.99, tuned on FEVER dev); headline numbers unchanged.

**Changed**
- S0-3: `factcheck/relabel.py` exports blind sheets (100 stratified FEVER-dev claims, passes 1 and 2 in different orders) to `data/relabel/` and scores Cohen's κ with CIs into `docs/label_quality.md`. No labels entered yet.
- S0-5: a 9-slide demo deck with a limits slide (Slides artifact, outside the repo), plus `docs/demo-walkthrough.md` with timing, tested live commands and fallbacks.
- Tests: 53, all passing.

**Next**
- Fill `data/relabel/pass1.csv`; fill `pass2.csv` several days later; run `python -m factcheck.relabel score`.
- Relevance gate for contradictions (tune on dev).
- Replace the deck placeholders: presenter name, course and group.

## 2026-11-25
**Reproduction run**
- SSD re-mounted (APFS, 401 GB free). `pytest`: 53 passed.
- Re-ran FEVER dev, FEVER test and Climate-FEVER evaluations. Every metric, CI and threshold is identical to 2026-11-14; only run dates and throughput timings changed (NLI 78–105 pairs/s, still above 5, so no dev cap).
- `check()` Eiffel Tower / Berlin example: REFUTES (0.998), span `[23, 29]` = "Berlin".

**Next**
- Unchanged: relabel study (S0-3) and a relevance gate for contradictions.
