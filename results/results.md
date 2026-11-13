# Results: fever-dev

*Generated 2026-11-14 by `python -m factcheck.run`. Bootstrap: 1,000 resamples of claims, seed 13, percentile 95% CI. Each number describes one system on one split; none of them is comparable with FEVER or AVeriTeC leaderboards.*

## Run facts
- Split: **fever-dev**, claims evaluated: **397** (no cap; throughput was above 5 pairs/s)
- Labels: FEVER crowd labels, 1 label per claim as released. No re-annotation by this project.
- NLI throughput: **110.3 claim-evidence pairs/s** on CPU (arm64 Darwin, batch 32, cross-encoder/nli-deberta-v3-xsmall). Retrieval: 74 queries/s.
- Thresholds used for the tuned system: SUPPORTS if max P(entailment) ≥ 0.95, REFUTES if max P(contradiction) ≥ 0.99, else NOT ENOUGH INFO. Tuned for macro-F1 on **FEVER dev only**.
- Claims flagged as opinion (predicted NOT ENOUGH INFO without checking): 3

## 1. Label quality (macro-F1 over 3 classes)
| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| NLI system, thresholds 0.5/0.5 (untuned): macro-F1 | 0.514 | [0.469, 0.556] | 397 | fever-dev | FEVER crowd labels, 1 label per claim as released |
| NLI system, thresholds 0.5/0.5 (untuned): accuracy | 0.572 | [0.521, 0.620] | 397 | fever-dev | FEVER crowd labels, 1 label per claim as released |
| NLI system, thresholds 0.95/0.99 (tuned on FEVER dev): macro-F1 | 0.635 | [0.590, 0.684] | 397 | fever-dev | FEVER crowd labels, 1 label per claim as released |
| NLI system, thresholds 0.95/0.99 (tuned on FEVER dev): accuracy | 0.630 | [0.584, 0.680] | 397 | fever-dev | FEVER crowd labels, 1 label per claim as released |
| Majority class (from FEVER train): macro-F1 | 0.160 | [0.143, 0.177] | 397 | fever-dev | FEVER crowd labels, 1 label per claim as released |
| Majority class (from FEVER train): accuracy | 0.315 | [0.272, 0.360] | 397 | fever-dev | FEVER crowd labels, 1 label per claim as released |
| Claim-only TF-IDF + LR (trained on FEVER train): macro-F1 | 0.403 | [0.349, 0.446] | 397 | fever-dev | FEVER crowd labels, 1 label per claim as released |
| Claim-only TF-IDF + LR (trained on FEVER train): accuracy | 0.406 | [0.353, 0.451] | 397 | fever-dev | FEVER crowd labels, 1 label per claim as released |

### NLI system, thresholds 0.5/0.5 (untuned)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.827 | 0.648 | 0.726 | 125 |
| REFUTES | 0.478 | 0.943 | 0.635 | 140 |
| NOT ENOUGH INFO | 0.609 | 0.106 | 0.181 | 132 |

Confusion matrix (rows = gold, columns = predicted):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 81 | 39 | 5 |
| REFUTES | 4 | 132 | 4 |
| NOT ENOUGH INFO | 13 | 105 | 14 |

### NLI system, thresholds 0.95/0.99 (tuned on FEVER dev)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.837 | 0.656 | 0.735 | 125 |
| REFUTES | 0.589 | 0.686 | 0.634 | 140 |
| NOT ENOUGH INFO | 0.529 | 0.545 | 0.537 | 132 |

Confusion matrix (rows = gold, columns = predicted):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 82 | 18 | 25 |
| REFUTES | 5 | 96 | 39 |
| NOT ENOUGH INFO | 11 | 49 | 72 |

### Majority class (from FEVER train)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.315 | 1.000 | 0.479 | 125 |
| REFUTES | 0.000 | 0.000 | 0.000 | 140 |
| NOT ENOUGH INFO | 0.000 | 0.000 | 0.000 | 132 |

Confusion matrix (rows = gold, columns = predicted):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 125 | 0 | 0 |
| REFUTES | 140 | 0 | 0 |
| NOT ENOUGH INFO | 132 | 0 | 0 |

### Claim-only TF-IDF + LR (trained on FEVER train)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTS | 0.323 | 0.256 | 0.286 | 125 |
| REFUTES | 0.611 | 0.493 | 0.545 | 140 |
| NOT ENOUGH INFO | 0.324 | 0.455 | 0.379 | 132 |

Confusion matrix (rows = gold, columns = predicted):

| gold \ pred | SUPPORTS | REFUTES | NEI |
|---|---|---|---|
| SUPPORTS | 32 | 21 | 72 |
| REFUTES | 18 | 69 | 53 |
| NOT ENOUGH INFO | 49 | 23 | 60 |

## 2. Retrieval (BM25, scored separately from the label)
Hit rate: share of SUPPORTS/REFUTES claims with at least one gold evidence sentence in the top k. NOT ENOUGH INFO claims have no gold evidence and are excluded.

| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| recall@1 | 0.525 | [0.464, 0.585] | 265 | fever-dev | FEVER crowd labels, 1 label per claim as released |
| recall@3 | 0.713 | [0.660, 0.766] | 265 | fever-dev | FEVER crowd labels, 1 label per claim as released |
| recall@5 | 0.808 | [0.758, 0.853] | 265 | fever-dev | FEVER crowd labels, 1 label per claim as released |

## 3. FEVER-style strict score
Label correct and, for SUPPORTS/REFUTES, a gold sentence in the top 5. This is a single-sentence approximation of the official FEVER score (which needs a complete evidence set), so it is not comparable with published FEVER scores.

| Metric | Value | 95% CI | n | Split | Annotators |
|---|---|---|---|---|---|
| FEVER-style strict score (tuned system) | 0.554 | [0.506, 0.605] | 397 | fever-dev | FEVER crowd labels, 1 label per claim as released |

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

- **correct**: `Veeru Devgan is from India.` (gold SUPPORTS, predicted SUPPORTS)
  - deciding evidence: `Veeru Devgan is an Indian stunt and action choreographer and film director in Bollywood.` (Veeru_Devgan:0)
- **correct**: `Ed Wood is not a movie.` (gold REFUTES, predicted REFUTES)
  - deciding evidence: `Ed Wood (film): The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau.` (Ed_Wood_-LRB-film-RRB-:1)
  - wrong spans: none
- **correct**: `Rage Against the Machine performed at concert halls and arenas.` (gold NOT ENOUGH INFO, predicted NOT ENOUGH INFO)
  - deciding evidence: `The same year, Rage Against the Machine announced a reunion and performed together for the first time in seven years at the Coachella Valley Music and Arts Festival in April 2007.` (Rage_Against_the_Machine:19)
- **wrong**: `Arizona is a mass-marketed beverage.` (gold REFUTES, predicted NOT ENOUGH INFO)
  - deciding evidence: `For atoms, the protons and neutrons of the nucleus account for almost all of the mass, and the atomic mass measured in u has nearly the same value as the mass number.` (Atomic_mass:3)
- **wrong**: `Fabian Nicieza was born December 31, 1961.` (gold SUPPORTS, predicted REFUTES)
  - deciding evidence: `Howard Gordon (born March 31, 1961) is an American television writer and producer.` (Howard_Gordon:0)
  - wrong spans: `Fabian Nicieza`, `December`
- **wrong**: `Raees (film) features a Pakistani actress in a lead role.` (gold NOT ENOUGH INFO, predicted REFUTES)
  - deciding evidence: `Pure (2009 film): Lead actress Alicia Vikander won the Swedish Guldbagge Award for Best Actress for her role in this film.` (Pure_-LRB-2009_film-RRB-:4)
  - wrong spans: `Raees`, `Pakistani`

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
