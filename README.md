# factcheck-bench: a CPU-only claim checker

A one-day prototype that takes a sentence and does four things:
- retrieves evidence from a fixed pool;
- labels the claim **SUPPORTS**, **REFUTES** or **NOT ENOUGH INFO**;
- marks the words that look wrong, as character offsets;
- skips opinions instead of checking them.

> "The Eiffel Tower is in Berlin" + evidence "The Eiffel Tower is in Paris" → `REFUTES`, wrong span `[23, 29]` = "Berlin".

A verdict means **"supported or refuted by the evidence we retrieved"**, not "true in the world". The evidence comes from a fixed pool, not the live internet, so NOT ENOUGH INFO means "not in our pool".

## Where this fits
This repo is the **claim checker**, one of three pipelines in the team's quote-attribution project:

1. **Quote tracing** (another team) finds copies of a quote online and picks the oldest.
2. **Confidence scoring** (another team) turns evidence into a calibrated confidence score.
3. **Claim checker** (this repo) decides whether a passage supports or refutes a claim, and where the claim goes wrong.

The other pipelines call one function:

```python
from factcheck.run import check

check("The Eiffel Tower is in Berlin", evidence=["The Eiffel Tower is in Paris."])
# {'label': 'REFUTES',
#  'scores': {'SUPPORTS': 0.0003, 'REFUTES': 0.9976, 'NOT ENOUGH INFO': 0.0021},
#  'evidence_used': [{'text': 'The Eiffel Tower is in Paris.', 'source_id': 'provided:0',
#                     'scores': {...}, 'decisive': True}],
#  'wrong_spans': [[23, 29]],
#  'is_opinion': False,
#  'schema_version': '1.0'}
```

- When the quote pipeline passes the passage it found as `evidence`, **retrieval is skipped**. That makes this repo the "does the new usage match the original passage" step. Without `evidence`, the claim is checked against the top 5 BM25 sentences from the pool.
- `scores` are the raw NLI softmax probabilities for the deciding evidence sentence. They are **uncalibrated** and are **not a confidence score**; turning them into one is the confidence pipeline's job.
- `wrong_spans` are `[start, end)` offsets into the claim string. They are filled only when the label is REFUTES.
- If the claim is flagged as an opinion, it isn't checked: `label` is NOT ENOUGH INFO, `is_opinion` is true, `scores` is `None`, and `evidence_used` is empty.
- `check_text(text, evidence=None)` splits free text into claims first and adds `claim_start` / `claim_end` to each result.

## What was built
| Stage | How | File |
|---|---|---|
| Ingest and validate | FEVER claims with gold evidence text (`copenlu/fever_gold_evidence`) and Climate-FEVER, mapped to schema v1 and validated | `factcheck/ingest.py` |
| Preprocess | Offset-preserving normalisation, regex claim splitter, rule-based opinion filter, de-duplication (exact hash, then 3-gram Jaccard), split by `source_id` (seed 13) | `factcheck/preprocess.py` |
| Retrieve | `rank_bm25` BM25Okapi over 30,221 sentences, using our own tokeniser, with page titles prefixed | `factcheck/retrieve.py` |
| Verdict | `cross-encoder/nli-deberta-v3-xsmall`, inference only, label order read from `model.config.id2label`; thresholds chosen on dev | `factcheck/verdict.py` |
| Spans | Rules: numbers, dates and capitalised names in the claim that are missing from the deciding evidence sentence | `factcheck/spans.py` |
| Evaluate | Macro-F1, per-class P/R, confusion matrix, recall@k, FEVER-style strict score, span character F1, bootstrap 95% CIs, majority and claim-only baselines | `factcheck/evaluate.py`, `factcheck/run.py` |

Docs: [data card](docs/data_card.md), [preprocessing log](docs/preprocessing-log.md), [demo walkthrough](docs/demo-walkthrough.md), [decisions](DECISIONS.md), [work log](WORKLOG.md).

## Headline results
The full tables (per-class P/R, confusion matrices, examples and caveats) are in [`results/results.md`](results/results.md) (FEVER dev), [`results/results-fever-test.md`](results/results-fever-test.md) and [`results/results-climate-test.md`](results/results-climate-test.md). Bootstrap: 1,000 resamples, 95% percentile CI. The labels are the released crowd labels; this project did no re-annotation.

| Metric (one system each) | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| Label macro-F1, NLI system (thresholds from dev) | 0.647 | [0.601, 0.690] | 396 | FEVER test | FEVER crowd |
| Label macro-F1, claim-only TF-IDF + LR | 0.431 | [0.381, 0.475] | 396 | FEVER test | FEVER crowd |
| Label macro-F1, majority class | 0.162 | [0.144, 0.178] | 396 | FEVER test | FEVER crowd |
| Retrieval recall@5 | 0.836 | [0.792, 0.880] | 250 | FEVER test (S/R claims) | FEVER crowd |
| Label macro-F1, NLI system (thresholds from FEVER dev) | 0.324 | [0.299, 0.351] | 1,351 | Climate-FEVER | Climate-FEVER crowd |
| Label macro-F1, claim-only TF-IDF + LR | 0.332 | [0.307, 0.357] | 1,351 | Climate-FEVER | Climate-FEVER crowd |
| Span char F1, rule only, given gold evidence | 0.856 | [0.640, 1.000] | 20 | hand-built | 1 (Claude, not a person) |

How to read these:
- **The NLI system beats the claim-only baseline on FEVER, but not on Climate-FEVER.** The FEVER thresholds don't transfer to climate claims, and the Climate-FEVER pool is small and noisy.
- **The claim-only baseline sits far above the majority class on FEVER** (0.431 vs 0.162). Claim wording alone leaks the label, so beating the majority class is not enough.
- **The main error on FEVER** is NOT ENOUGH INFO claims predicted REFUTES. The NLI model reads "contradiction" into sentences that are on the same topic but irrelevant.
- **None of these numbers are comparable with FEVER or AVeriTeC leaderboards.** The sample is balanced, the pool is seeded with gold evidence, and the FEVER-style score is a single-sentence approximation.
- **Throughput:** 76–110 claim-evidence pairs/s on an Apple-silicon CPU. That is above 5 pairs/s, so dev evaluation was **not capped**.

## What was not built
- **Self-relabel agreement study (S0-3).** The blind sheets and scoring are ready (`data/relabel/`, `python -m factcheck.relabel`), but no labels have been entered yet, so no agreement number is reported. The two passes need days between them (S1-4).
- **Slides (S0-5).** The 9-slide demo deck with a limits slide is kept outside the repo as a Slides artifact. Its timing, live commands and fallbacks are in [docs/demo-walkthrough.md](docs/demo-walkthrough.md).
- **Full Wikipedia retrieval.** `--full-wiki` exists but stops with a message. The dump is about 1.7 GB zipped and 7.25 GB expanded, and needs approval before downloading.
- **Negation, verb swaps and common-noun swaps in spans.** The span rules don't handle these. "Einstein did **not** win the Nobel Prize" and "for the top **baseball** clubs" get the right label but no span.
- **No training or fine-tuning**, no GPU code, no LLM APIs, no web UI, no calibrated confidence.
- **No labels of the user's own.** None were provided, and none were invented.

## Setup
```bash
cp config.example.toml config.local.toml       # set data_dir to a folder on the external drive
# or: export FACTCHECK_DATA=/path/to/drive/folder
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"   # deps in pyproject.toml
.venv/bin/python -m factcheck.ingest       # ~110 MB of data into <data_dir>/raw
.venv/bin/python -m factcheck.preprocess   # splits + docs/preprocessing-log.md
.venv/bin/pytest                           # check() tests download the model (~280 MB)
.venv/bin/python -m factcheck.run --split dev                    # tunes thresholds -> results/results.md
.venv/bin/python -m factcheck.run --split test                   # uses dev thresholds
.venv/bin/python -m factcheck.run --split test --dataset climate # out-of-domain, eval only
```
Everything large (datasets, BM25 index, NLI cache, model weights, and here also the venv) lives under the data directory. The HF and pip caches are pointed there by `factcheck/config.py`.

## Timeline (real)
- **2026-10-27 to 2026-11-02:** research and planning notes (implementation brief, schema draft, dataset matrix, backlog). These aren't in this repo.
- **Planned:** Sprint 0 on 2026-11-02 to 11-04, Sprint 1 on 2026-11-06 to 11-07.
- **Actual:** all code, data processing and evaluation here were built in **one session on 2026-11-14**, a week after the planned Sprint 1 end. The commit dates are the real ones.

## Licences
- Code: not yet chosen by the team.
- FEVER-derived data: CC BY-SA 3.0 (the dataset card also lists GPL-3.0).
- Climate-FEVER: licence listed as **unknown**, so it is used for evaluation only and not redistributed.
- Model: Apache-2.0.
- No data or weights are committed.
