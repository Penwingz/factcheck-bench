# Demo walkthrough (about 5 minutes)

The deck has 9 slides; each one's speaker notes hold its script. This page gives the timing, the commands for the live part, and fallbacks if the live part fails. All numbers on the slides come from `results/*.json` as of 2026-11-14.

## Timing
| Time | Slide | Point to land |
|---|---|---|
| 0:00 | 1 Title | One pipeline of three, CPU only, no training |
| 0:20 | 2 Where this fits | A verdict means "supported or refuted by the evidence we used" |
| 0:50 | 3 Five steps | The NLI model never saw FEVER, so there's no leakage |
| 1:25 | 4 Live example | Run the Eiffel/Berlin command below (about 15 s, including the model load) |
| 2:10 | 5 Data | Split by source, nothing tuned on test, no labels invented |
| 2:40 | 6 Results | Compare with the claim-only baseline, not the majority class; Climate gives no gain |
| 3:30 | 7 Three numbers | Stages are scored separately; the biggest error is NEI read as REFUTES |
| 4:05 | 8 Limits | The slide to remember |
| 4:40 | 9 Next | One-call integration; next steps |

## Before presenting
```bash
# the external drive must be mounted (data, model cache and venv live there)
ls "$(.venv/bin/python -c 'from factcheck import config; print(config.data_dir())')"
.venv/bin/pytest -q          # sanity check, about 10 s
```
Run one `check()` beforehand so the model is already in the OS file cache.

## Live commands
Evidence passed in (slide 4):
```bash
.venv/bin/python -c "
import json; from factcheck.run import check
print(json.dumps(check('The Eiffel Tower is in Berlin', ['The Eiffel Tower is in Paris.']), indent=1))"
```
Expected: `REFUTES`, `scores.REFUTES` = 0.9976, `wrong_spans` = `[[23, 29]]` ("Berlin").

Retrieval from the pool (optional, if there's time):
```bash
.venv/bin/python -c "
from factcheck.run import check
r = check('Jackie (2016 film) was directed by Peter Jackson.')
c = 'Jackie (2016 film) was directed by Peter Jackson.'
print(r['label'], [c[a:b] for a, b in r['wrong_spans']])"
```
Expected: `REFUTES ['Peter Jackson']`. The evidence comes from the pool (Pablo Larraín directed it).

An honest failure to show if asked "does retrieval always work?":
`check('Syracuse, New York, had a population of 145,170 according to the 2010 United States Census.')` returns NOT ENOUGH INFO. BM25 ranks a 2010 census sentence about Prescott, Arizona above the Syracuse one.

Opinion filter: `check('I think Paris is the nicest city in Europe.')` returns NOT ENOUGH INFO with `is_opinion: true` and `scores: None`.

## Fallback if the live run fails
Show slide 4 as it is. Its JSON was copied from a real run on 2026-11-14, and `tests/test_check.py::test_eiffel_tower_berlin_refuted_with_span` checks the same case.

## Likely questions
- **Why not GPT or another LLM?** The brief rules out paid APIs and training. A small NLI cross-encoder runs on a laptop and was never trained on FEVER.
- **Why is the claim-only baseline so high?** FEVER claims were written by people editing Wikipedia sentences, so words like "only" or "not" give the label away. That's why it's the baseline to beat.
- **Is 0.647 good?** Better than the shortcut on held-out FEVER, with no overlap between the CIs. It isn't comparable with leaderboard numbers (see Limits).
- **Agreement?** No study yet: the relabel sheets exist (`data/relabel/`), and the two passes need days between them.
