# Decisions

Each entry: date, decision, reason. Entries marked *(adopted)* were proposed in the planning notes on 2026-11-02 and confirmed in the build.

| Date | Decision | Why |
|---|---|---|
| 2026-11-02 *(adopted)* | Small evidence pool, not the full FEVER Wikipedia dump (about 1.7 GB zipped, 7.25 GB expanded). `--full-wiki` is a flag, off by default and not implemented. | Keeps the build to one day and the disk small |
| 2026-11-02 *(adopted)* | Verdict model `cross-encoder/nli-deberta-v3-xsmall`, inference only | Small enough for CPU. It was trained on SNLI and MNLI, so evaluating on FEVER doesn't leak training data. |
| 2026-11-14 | Take FEVER claims and evidence *text* from `copenlu/fever_gold_evidence` (`valid.jsonl`) | It has the sentence text for each gold pointer, so the wiki dump isn't needed. 6.5 MB. |
| 2026-11-14 | Evidence pool = every evidence sentence in `valid.jsonl`, plus 24,000 seeded sentences from `train.jsonl` (30,221 in total) | 6,221 sentences gave retrieval too little competition. Downside: the pool is still seeded with gold evidence, which is stated as a caveat. |
| 2026-11-14 | Stratified sample of 1,000 claims per label (seed 13) | Gives the three classes equal weight in macro-F1. Not FEVER's real label mix; stated as a caveat. |
| 2026-11-14 | `source_id` = the page of a claim's first evidence sentence (for NEI, the page of its retrieved sentence) | FEVER claims have no source document. The page keeps mutation pairs about the same entity in one split. |
| 2026-11-14 | NEI evidence in the source file is treated as retrieved context, not gold | The file attaches retrieved sentences to NEI claims. Counting them as gold would distort recall@k. |
| 2026-11-14 | De-dup: exact duplicates merge their gold evidence; near duplicates (word 3-gram Jaccard ≥ 0.9) are dropped only if the labels agree | FEVER repeats a claim once per evidence set. A first pass at 0.8 without the label check dropped real mutation pairs ("Queen Victoria" / "Queen Elizabeth"). |
| 2026-11-14 | Opinion-flagged claims stay in the evaluation and are predicted NOT ENOUGH INFO | Dropping them would hide the filter's false positives. 3 FEVER-dev claims are affected. |
| 2026-11-14 | Index and verify each sentence with its page title prefixed | Many Wikipedia sentences start with "He" or "It"; without the title they can't be matched |
| 2026-11-14 | Aggregation: SUPPORTS if max P(entail) ≥ τs, else REFUTES if max P(contra) ≥ τr, else NEI; if both pass, the larger wins | Simple, and leaves two numbers to tune |
| 2026-11-14 | τs = 0.95 and τr = 0.99, grid-searched for macro-F1 on FEVER dev only. After the first run the grid was extended past 0.95, because the optimum sat on the edge. | NLI over-predicts contradiction for same-topic but irrelevant sentences, so strict thresholds help. Untuned 0.5/0.5 is reported next to the tuned numbers. |
| 2026-11-14 | FEVER test and Climate-FEVER were each run once, after the thresholds were frozen. Nothing was changed in response to them. | Rule: never tune on test |
| 2026-11-14 | Spans are computed only for REFUTES predictions, against the deciding sentence. They're scored on a 20-claim set built by Claude (`annotator=claude-handbuilt`). | The public data has no span labels. The set includes 3 known-miss cases (negation, common-noun swap) so the score isn't flattered. |
| 2026-11-14 | The "every REFUTES row has a span" validation applies only to the hand-built span set | FEVER and Climate-FEVER have no spans, so the rule would reject every REFUTES row |
| 2026-11-14 | Climate-FEVER DISPUTED claims (154) are dropped, not mapped | The schema has three labels; DISPUTED isn't NEI |
| 2026-11-14 | Python 3.12 venv on the external SSD, symlinked as `.venv` (gitignored) | The system Python is 3.14, which is too new to trust for torch wheels. The venv is about 1 GB and the internal disk is short of space. |
| 2026-11-14 | Git identity for this repo only: Penwingz with the GitHub noreply address | rules.md names the main committer; the global identity is left alone for other projects |
| 2026-11-14 | `pyproject.toml` with pytest `pythonpath = ["."]` | The brief's command is bare `pytest`, which couldn't import the package without it |
| 2026-11-14 | Bootstrap CIs on per-class P/R/F1 and span P/R/F1, from one resampling pass (`bootstrap_many`) | Rule 3 (every metric has a CI); one pass keeps the runtime low |
| 2026-11-14 | Relabel sheets show the gold sentence for SUPPORTS/REFUTES and the top BM25 sentence for NEI, with no label column filled | NEI has no gold sentence. Showing the evidence FEVER associated with each claim keeps agreement with FEVER meaningful. |
| 2026-11-14 | The deck chart is dot-and-whisker, with orange for this system, grey for baselines, and every row labelled directly | Colour never carries meaning alone. The palette was checked with the dataviz validator: normal-vision ΔE 18.5, CVD ΔE 14.5, contrast ≥ 3:1. The grey is deliberately neutral. |
