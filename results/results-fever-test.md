# Results: fever-test

*Generated 2026-11-14 by `python -m factcheck.run`. Bootstrap: 1,000 resamples of claims, seed 13, percentile 95% CI. Each number describes one system on one split; none of them is comparable with FEVER or AVeriTeC leaderboards.*

## Run facts
- Split: **fever-test**, claims evaluated: **396** (no cap; throughput was above 5 pairs/s)
- Labels: FEVER crowd labels, 1 label per claim as released. No re-annotation by this project.
- NLI throughput: **86.8 claim-evidence pairs/s** on CPU (arm64 Darwin, batch 32, cross-encoder/nli-deberta-v3-xsmall). Retrieval: 74 queries/s.
- Thresholds used for the tuned system: SUPPORTS if max P(entailment) ≥ 0.95, REFUTES if max P(contradiction) ≥ 0.99, else NOT ENOUGH INFO. Tuned for macro-F1 on **FEVER dev only**.
- Claims flagged as opinion (predicted NOT ENOUGH INFO without checking): 0

## 1. Label quality (macro-F1 over 3 classes)
| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| NLI system, thresholds 0.5/0.5 (untuned): macro-F1 | 0.536 | [0.488, 0.579] | 396 | fever-test | FEVER crowd labels, 1 label per claim as released |
| NLI system, thresholds 0.5/0.5 (untuned): accuracy | 0.563 | [0.518, 0.614] | 396 | fever-test | FEVER crowd labels, 1 label per claim as released |
| NLI system, thresholds 0.95/0.99 (tuned on FEVER dev): macro-F1 | 0.647 | [0.601, 0.690] | 396 | fever-test | FEVER crowd labels, 1 label per claim as released |
| NLI system, thresholds 0.95/0.99 (tuned on FEVER dev): accuracy | 0.644 | [0.598, 0.687] | 396 | fever-test | FEVER crowd labels, 1 label per claim as released |
| Majority class (from FEVER train): macro-F1 | 0.162 | [0.144, 0.178] | 396 | fever-test | FEVER crowd labels, 1 label per claim as released |
| Majority class (from FEVER train): accuracy | 0.321 | [0.275, 0.364] | 396 | fever-test | FEVER crowd labels, 1 label per claim as released |
| Claim-only TF-IDF + LR (trained on FEVER train): macro-F1 | 0.431 | [0.381, 0.475] | 396 | fever-test | FEVER crowd labels, 1 label per claim as released |
| Claim-only TF-IDF + LR (trained on FEVER train): accuracy | 0.432 | [0.381, 0.475] | 396 | fever-test | FEVER crowd labels, 1 label per claim as released |

### NLI system, thresholds 0.5/0.5 (untuned)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.811 | 0.677 | 0.738 | 127 |
| REFUTES | 0.439 | 0.911 | 0.593 | 123 |
| NOT ENOUGH INFO | 0.714 | 0.171 | 0.276 | 146 |

Confusion matrix (rows = gold, columns = predicted):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 86 | 36 | 5 |
| REFUTES | 6 | 112 | 5 |
| NOT ENOUGH INFO | 14 | 107 | 25 |

### NLI system, thresholds 0.95/0.99 (tuned on FEVER dev)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.798 | 0.622 | 0.699 | 127 |
| REFUTES | 0.577 | 0.642 | 0.608 | 123 |
| NOT ENOUGH INFO | 0.606 | 0.664 | 0.634 | 146 |

Confusion matrix (rows = gold, columns = predicted):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 79 | 21 | 27 |
| REFUTES | 8 | 79 | 36 |
| NOT ENOUGH INFO | 12 | 37 | 97 |

### Majority class (from FEVER train)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.321 | 1.000 | 0.486 | 127 |
| REFUTES | 0.000 | 0.000 | 0.000 | 123 |
| NOT ENOUGH INFO | 0.000 | 0.000 | 0.000 | 146 |

Confusion matrix (rows = gold, columns = predicted):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 127 | 0 | 0 |
| REFUTES | 123 | 0 | 0 |
| NOT ENOUGH INFO | 146 | 0 | 0 |

### Claim-only TF-IDF + LR (trained on FEVER train)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.388 | 0.299 | 0.338 | 127 |
| REFUTES | 0.624 | 0.431 | 0.510 | 123 |
| NOT ENOUGH INFO | 0.376 | 0.548 | 0.446 | 146 |

Confusion matrix (rows = gold, columns = predicted):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 38 | 11 | 78 |
| REFUTES | 15 | 53 | 55 |
| NOT ENOUGH INFO | 45 | 21 | 80 |

## 2. Retrieval (BM25, scored separately from the label)
Hit rate: share of SUPPORTS/REFUTES claims with at least one gold evidence sentence in the top k. NOT ENOUGH INFO claims have no gold evidence and are excluded.

| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| recall@1 | 0.556 | [0.500, 0.616] | 250 | fever-test | FEVER crowd labels, 1 label per claim as released |
| recall@3 | 0.760 | [0.708, 0.812] | 250 | fever-test | FEVER crowd labels, 1 label per claim as released |
| recall@5 | 0.836 | [0.792, 0.880] | 250 | fever-test | FEVER crowd labels, 1 label per claim as released |

## 3. FEVER-style strict score
Label correct and, for SUPPORTS/REFUTES, a gold sentence in the top 5. This is a single-sentence approximation of the official FEVER score (which needs a complete evidence set), so it is not comparable with published FEVER scores.

| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| FEVER-style strict score (tuned system) | 0.591 | [0.545, 0.639] | 396 | fever-test | FEVER crowd labels, 1 label per claim as released |

## 4. Wrong-span character F1 (hand-built set)
The public datasets have no span labels, so spans are scored only on `data/span_gold_handbuilt.jsonl`: 20 claims (15 REFUTES, the rest SUPPORTS). Three REFUTES cases are negation or common-noun swaps, which the rules are known not to handle. Treat this as a smoke test, not a benchmark.

| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| Span char F1, rule only, given gold evidence | 0.856 | [0.640, 1.000] | 20 | handbuilt | 1 (Claude, hand-built; not a human annotator) |
| Span char F1, end-to-end check() with gold evidence | 0.647 | [0.325, 0.857] | 20 | handbuilt | 1 (Claude, hand-built; not a human annotator) |

- Rule only: precision 1.000, recall 0.748.
- End to end: precision 1.000, recall 0.478; spans appear only when the label is REFUTES. Label accuracy on these 20 claims: 0.750.

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

- **correct**: `Luke Cage evolved into a person whose skills were leased out.` (gold NOT ENOUGH INFO, predicted NOT ENOUGH INFO)
  - deciding evidence: `A secretary or personal assistant is a person whose work consists of supporting management, including executives, using a variety of project management, communication, or organizational skills.` (Secretary:0)
- **correct**: `Ian Brennan is a writer.` (gold SUPPORTS, predicted SUPPORTS)
  - deciding evidence: `Ian Brennan (writer): Ian Brennan (born April 23, 1978) is a television writer, actor, producer and director.` (Ian_Brennan_-LRB-writer-RRB-:0)
- **correct**: `Garden State was at the dead body.` (gold NOT ENOUGH INFO, predicted NOT ENOUGH INFO)
  - deciding evidence: `Garden State refers to the official nicknames of the state of New Jersey in the United States and the state of Victoria in Australia.` (Garden_State:0)
- **wrong**: `Tijuana is in Mexico.` (gold REFUTES, predicted SUPPORTS)
  - deciding evidence: `As an industrial and financial center of Mexico, Tijuana exerts a strong influence on economics, education, culture, art, and politics.` (Tijuana:1)
- **wrong**: `Jeong Hyeong-don is from South Korea.` (gold NOT ENOUGH INFO, predicted SUPPORTS)
  - deciding evidence: `Jeong Hyeong-don (born February 7, 1978) is a South Korean comedian and entertainer under FNC Entertainment.` (Jeong_Hyeong-don:0)
- **wrong**: `Advertising is a closed sourced message.` (gold NOT ENOUGH INFO, predicted REFUTES)
  - deciding evidence: `Advertising is an audio or visual form of marketing communication that employs an openly sponsored, nonpersonal message to promote or sell a product, service or idea.` (Advertising:0)
  - wrong spans: none

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
