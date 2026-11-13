# Results: climate-test

*Generated 2026-11-14 by `python -m factcheck.run`. Bootstrap: 1,000 resamples of claims, seed 13, percentile 95% CI. Each number describes one system on one split; none of them is comparable with FEVER or AVeriTeC leaderboards.*

## Run facts
- Split: **climate-test**, claims evaluated: **1351** (no cap; throughput was above 5 pairs/s)
- Labels: Climate-FEVER crowd, 5 votes per evidence sentence. No re-annotation by this project.
- NLI throughput: **82.0 claim-evidence pairs/s** on CPU (arm64 Darwin, batch 32, cross-encoder/nli-deberta-v3-xsmall). Retrieval: 241 queries/s.
- Thresholds used for the tuned system: SUPPORTS if max P(entailment) ≥ 0.95, REFUTES if max P(contradiction) ≥ 0.99, else NOT ENOUGH INFO. Tuned for macro-F1 on **FEVER dev only**.
- Claims flagged as opinion (predicted NOT ENOUGH INFO without checking): 25

## 1. Label quality (macro-F1 over 3 classes)
| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| NLI system, thresholds 0.5/0.5 (untuned): macro-F1 | 0.336 | [0.310, 0.361] | 1351 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |
| NLI system, thresholds 0.5/0.5 (untuned): accuracy | 0.346 | [0.320, 0.372] | 1351 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |
| NLI system, thresholds 0.95/0.99 (tuned on FEVER dev): macro-F1 | 0.324 | [0.299, 0.351] | 1351 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |
| NLI system, thresholds 0.95/0.99 (tuned on FEVER dev): accuracy | 0.392 | [0.366, 0.416] | 1351 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |
| Majority class (from FEVER train): macro-F1 | 0.215 | [0.207, 0.223] | 1351 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |
| Majority class (from FEVER train): accuracy | 0.476 | [0.451, 0.503] | 1351 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |
| Claim-only TF-IDF + LR (trained on FEVER train): macro-F1 | 0.332 | [0.307, 0.357] | 1351 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |
| Claim-only TF-IDF + LR (trained on FEVER train): accuracy | 0.353 | [0.329, 0.378] | 1351 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |

### NLI system, thresholds 0.5/0.5 (untuned)

Per class, value [95% CI]; n = 1351, split climate-test, annotators: Climate-FEVER crowd, 5 votes per evidence sentence.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.775 [0.698, 0.851] | 0.134 [0.108, 0.161] | 0.228 [0.188, 0.269] | 643 |
| REFUTES | 0.262 [0.229, 0.296] | 0.671 [0.616, 0.733] | 0.377 [0.336, 0.417] | 240 |
| NOT ENOUGH INFO | 0.351 [0.315, 0.389] | 0.470 [0.427, 0.512] | 0.402 [0.367, 0.439] | 468 |

Confusion matrix, counts (rows = gold, columns = predicted; n = 1351):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 86 | 218 | 339 |
| REFUTES | 12 | 161 | 67 |
| NOT ENOUGH INFO | 13 | 235 | 220 |

### NLI system, thresholds 0.95/0.99 (tuned on FEVER dev)

Per class, value [95% CI]; n = 1351, split climate-test, annotators: Climate-FEVER crowd, 5 votes per evidence sentence.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.844 [0.744, 0.941] | 0.059 [0.042, 0.077] | 0.110 [0.079, 0.142] | 643 |
| REFUTES | 0.474 [0.400, 0.562] | 0.267 [0.215, 0.324] | 0.341 [0.283, 0.402] | 240 |
| NOT ENOUGH INFO | 0.365 [0.339, 0.392] | 0.912 [0.889, 0.938] | 0.521 [0.493, 0.549] | 468 |

Confusion matrix, counts (rows = gold, columns = predicted; n = 1351):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 38 | 34 | 571 |
| REFUTES | 3 | 64 | 173 |
| NOT ENOUGH INFO | 4 | 37 | 427 |

### Majority class (from FEVER train)

Per class, value [95% CI]; n = 1351, split climate-test, annotators: Climate-FEVER crowd, 5 votes per evidence sentence.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.476 [0.451, 0.503] | 1.000 [1.000, 1.000] | 0.645 [0.622, 0.670] | 643 |
| REFUTES | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 240 |
| NOT ENOUGH INFO | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 468 |

Confusion matrix, counts (rows = gold, columns = predicted; n = 1351):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 643 | 0 | 0 |
| REFUTES | 240 | 0 | 0 |
| NOT ENOUGH INFO | 468 | 0 | 0 |

### Claim-only TF-IDF + LR (trained on FEVER train)

Per class, value [95% CI]; n = 1351, split climate-test, annotators: Climate-FEVER crowd, 5 votes per evidence sentence.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.491 [0.428, 0.556] | 0.170 [0.141, 0.200] | 0.252 [0.215, 0.292] | 643 |
| REFUTES | 0.263 [0.213, 0.307] | 0.350 [0.290, 0.410] | 0.301 [0.250, 0.345] | 240 |
| NOT ENOUGH INFO | 0.351 [0.320, 0.384] | 0.607 [0.564, 0.652] | 0.444 [0.412, 0.477] | 468 |

Confusion matrix, counts (rows = gold, columns = predicted; n = 1351):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 109 | 129 | 405 |
| REFUTES | 35 | 84 | 121 |
| NOT ENOUGH INFO | 78 | 106 | 284 |

## 2. Retrieval (BM25, scored separately from the label)
Hit rate: share of SUPPORTS/REFUTES claims with at least one gold evidence sentence in the top k. NOT ENOUGH INFO claims have no gold evidence and are excluded.

| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| recall@1 | 0.266 | [0.238, 0.294] | 883 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |
| recall@3 | 0.430 | [0.397, 0.461] | 883 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |
| recall@5 | 0.492 | [0.458, 0.525] | 883 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |

## 3. FEVER-style strict score
Label correct and, for SUPPORTS/REFUTES, a gold sentence in the top 5. This is a single-sentence approximation of the official FEVER score (which needs a complete evidence set), so it is not comparable with published FEVER scores.

| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| FEVER-style strict score (tuned system) | 0.364 | [0.338, 0.389] | 1351 | climate-test | Climate-FEVER crowd, 5 votes per evidence sentence |

## 4. Wrong-span character F1 (hand-built set)
The public datasets have no span labels, so spans are scored only on `data/span_gold_handbuilt.jsonl`: 20 claims (15 REFUTES, the rest SUPPORTS). Three REFUTES cases are negation or common-noun swaps, which the rules are known not to handle. Treat this as a smoke test, not a benchmark.

| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| Span char f1, rule only, given gold evidence | 0.856 | [0.640, 1.000] | 20 | handbuilt | 1 (Claude, hand-built; not a human annotator) |
| Span char precision, rule only, given gold evidence | 1.000 | [1.000, 1.000] | 20 | handbuilt | 1 (Claude, hand-built; not a human annotator) |
| Span char recall, rule only, given gold evidence | 0.748 | [0.471, 1.000] | 20 | handbuilt | 1 (Claude, hand-built; not a human annotator) |
| Span char f1, end-to-end check() with gold evidence | 0.647 | [0.325, 0.857] | 20 | handbuilt | 1 (Claude, hand-built; not a human annotator) |
| Span char precision, end-to-end check() with gold evidence | 1.000 | [1.000, 1.000] | 20 | handbuilt | 1 (Claude, hand-built; not a human annotator) |
| Span char recall, end-to-end check() with gold evidence | 0.478 | [0.194, 0.750] | 20 | handbuilt | 1 (Claude, hand-built; not a human annotator) |

End to end, spans appear only when the label is REFUTES. Label accuracy of check() on these 20 claims: 0.750.

| Claim | Gold label | Predicted | Gold span | Predicted spans |
|---|---|---|---|---|
| The Eiffel Tower is in Berlin. | REFUTES | REFUTES | Berlin | Berlin |
| Barack Obama was born in 1962. | REFUTES | NOT ENOUGH INFO | 1962 | - |
| Jackie (2016 film) was directed by Peter Jackson. | REFUTES | REFUTES | Peter Jackson | Peter Jackson |
| Mount Everest is 7,848 metres tall. | REFUTES | NOT ENOUGH INFO | 7,848 | - |
| The Beatles formed in Manchester in 1960. | REFUTES | REFUTES | Manchester | Manchester |
| World War II ended in 1949. | REFUTES | NOT ENOUGH INFO | 1949 | - |
| The Amazon River flows through Argentina. | REFUTES | NOT ENOUGH INFO | Argentina | - |
| Hamlet was written by Charles Dickens. | REFUTES | REFUTES | Charles Dickens | Charles Dickens |
| The Moon landing took place in July 1972. | REFUTES | REFUTES | 1972 | 1972 |
| Victoria Palace Theatre is in a place named after Queen Elizabeth. | REFUTES | NOT ENOUGH INFO | Elizabeth | - |
| Tokyo has a population of three million. | REFUTES | REFUTES | three | three |
| The CONCACAF Champions League is organized for the top baseball clubs. | REFUTES | REFUTES | baseball | - |
| Albert Einstein did not win the Nobel Prize. | REFUTES | REFUTES | did not | - |
| Steven Spielberg produced Schindler's List but did not direct it. | REFUTES | REFUTES | did not direct | - |
| Water boils at 50 degrees Celsius at sea level. | REFUTES | REFUTES | 50 | 50 |
| Marie Curie was born in Warsaw. | SUPPORTS | SUPPORTS | - | - |
| Paris is the capital of France. | SUPPORTS | SUPPORTS | - | - |
| Thriller was released in 1982. | SUPPORTS | SUPPORTS | - | - |
| Leonardo da Vinci painted the Mona Lisa. | SUPPORTS | SUPPORTS | - | - |
| Canberra is the capital of Australia. | SUPPORTS | SUPPORTS | - | - |

## 5. Worked examples (tuned system)

- **correct**: `The Great Barrier Reef is experiencing the most widespread bleaching ever recorded` (gold SUPPORTS, predicted SUPPORTS)
  - deciding evidence: `These temperatures have caused the most severe and widespread coral bleaching ever recorded in the Great Barrier reef.` (Coral bleaching:52)
- **correct**: `it’s not a pollutant that threatens human civilization.` (gold REFUTES, predicted REFUTES)
  - deciding evidence: `Global catastrophic risk: The report warned that the pollution crisis was exceeding "the envelope on the amount of pollution the Earth can carry" and “threatens the continuing survival of human societies”.` (Global catastrophic risk:104)
  - wrong spans: none
- **correct**: `Sea level rise has been slow and a constant, pre-dating industrialization` (gold REFUTES, predicted REFUTES)
  - deciding evidence: `Sea level rise: "Climate-change–driven accelerated sea-level rise detected in the altimeter era".` (Sea level rise:405)
  - wrong spans: none
- **wrong**: `Global warming is driving polar bears toward extinction` (gold SUPPORTS, predicted NOT ENOUGH INFO)
  - deciding evidence: `Natural predators of the bearded seal include polar bears, who rely on these seals as a major food source.` (Bearded seal:54)
- **wrong**: `The sun has gone into ‘lockdown’ which could cause freezing weather, earthquakes and famine, say scientists` (gold SUPPORTS, predicted NOT ENOUGH INFO)
  - deciding evidence: `A primary cause of the famine (one of the largest seen in the country) is that Ethiopia (and the surrounding Horn) was still recovering from the droughts which occurred in the mid-late 1970s.` (Famine:131)
- **wrong**: `The polar bear population has been growing.` (gold REFUTES, predicted NOT ENOUGH INFO)
  - deciding evidence: `Warnings about the future of the polar bear are often contrasted with the fact that worldwide population estimates have increased over the past 50 years and are relatively stable today.` (Polar bear:404)

## Caveats
- **Tuned on this split (dev only):** the tuned thresholds were chosen on FEVER dev, so FEVER-dev numbers for the tuned system are optimistic. The untuned 0.5/0.5 row is the fairer dev number; FEVER test and Climate-FEVER use the dev thresholds unchanged.
- **Gold-seeded pool:** the FEVER pool (30,221 sentences) contains every gold sentence plus distractors, so retrieval is far easier than over all of Wikipedia. NOT ENOUGH INFO means "not in our pool".
- **Balanced sample:** the 1,000-per-label sample does not match FEVER's real label mix.
- **Labels:** one released crowd label per claim; this project did no re-annotation and has no agreement study (S0-3 not done).
- **Claim-only baseline:** where it beats the majority class, claim wording alone predicts the label (a FEVER artefact). Compare the NLI system with it, not only with the majority class.
- **Scores are uncalibrated** softmax outputs, not confidence scores.
- **Spans:** rule-based, scored on a small Claude-built set. Negation, verb swaps and common-noun swaps are not handled.
- **Climate-FEVER** licence is listed as unknown; it is used for evaluation only. DISPUTED claims were dropped.
- **No leakage from training:** the NLI model was trained on SNLI and MNLI, not FEVER. Nothing was trained or fine-tuned here; thresholds are the only fitted values, plus the claim-only baseline trained on FEVER train.
