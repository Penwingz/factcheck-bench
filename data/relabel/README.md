# Self-relabel sheets (S0-3, S1-4)

`pass1.csv` and `pass2.csv` hold the same 100 FEVER-dev claims in different orders. The FEVER label is hidden.

**How to label**
1. Fill `pass1.csv` now. Put one label per row in the `label` column: `SUPPORTS`, `REFUTES` or `NOT ENOUGH INFO` (`S`, `R`, `N` also work). Use `notes` for anything unsure.
2. Judge **only from the evidence sentence shown**, not from what you know about the world:
   - SUPPORTS: the sentence says the claim is true.
   - REFUTES: the sentence says the claim is false.
   - NOT ENOUGH INFO: the sentence doesn't settle it.
3. Wait **several days**, without looking at pass 1, then fill `pass2.csv`.
4. Score it:
   `python -m factcheck.relabel score --pass1-date YYYY-MM-DD --pass2-date YYYY-MM-DD`
   This writes `docs/label_quality.md`.

The export won't overwrite a sheet that already has labels in it.
