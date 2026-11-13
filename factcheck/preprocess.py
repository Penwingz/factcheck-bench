"""Offset-preserving normalization, claim splitting, opinion filter, de-dup, split.

Usage: python -m factcheck.preprocess   (writes splits to the data dir and
docs/preprocessing-log.md)
"""
import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace

from . import config
from .ingest import Row, load, save

SEED = 13
RATIOS = (0.70, 0.15, 0.15)

# FEVER/PTB tokenizer escapes found in the Wikipedia sentences.
PTB_TOKENS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
              "-LCB-": "{", "-RCB-": "}", "``": '"', "''": '"', "-COLON-": ":"}
_NO_SPACE_BEFORE = set(".,;:!?)]}%'")
_NO_SPACE_AFTER = set("([{$")


class Normalized:
    """Normalized text plus, for every character, the raw span it came from."""

    def __init__(self, text: str, starts: list[int], ends: list[int]):
        self.text, self.starts, self.ends = text, starts, ends

    def to_raw(self, start: int, end: int) -> tuple[int, int]:
        """Map a [start, end) span in normalized text back to raw offsets."""
        if start >= end:
            raise ValueError("empty span")
        return self.starts[start], self.ends[end - 1]


def normalize(raw: str) -> Normalized:
    """NFKC, PTB-token un-escaping, whitespace collapse and FEVER-style
    space-before-punctuation removal, keeping a char-level map to `raw`."""
    chars: list[tuple[str, int, int]] = []
    i = 0
    while i < len(raw):
        for tok, rep in PTB_TOKENS.items():
            if raw.startswith(tok, i):
                chars += [(c, i, i + len(tok)) for c in rep]
                i += len(tok)
                break
        else:
            for c in unicodedata.normalize("NFKC", raw[i]):
                chars.append((" " if c.isspace() else c, i, i + 1))
            i += 1
    out: list[tuple[str, int, int]] = []
    for idx, ch in enumerate(chars):
        c = ch[0]
        if c == " ":
            nxt = next((x[0] for x in chars[idx + 1:] if x[0] != " "), None)
            if not out or out[-1][0] == " " or nxt is None:
                continue
            if nxt in _NO_SPACE_BEFORE or out[-1][0] in _NO_SPACE_AFTER:
                continue
        out.append(ch)
    return Normalized("".join(c for c, _, _ in out), [s for _, s, _ in out],
                      [e for _, _, e in out])


# Split after . ! ? followed by space and an uppercase letter or quote, but not
# after common abbreviations or single initials ("J. R. R. Tolkien").
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "inc", "ltd",
           "co", "no", "u.s", "u.k", "e.g", "i.e", "etc", "jan", "feb", "mar", "apr",
           "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_claims(text: str) -> list[tuple[int, int]]:
    """Return [start, end) offsets of sentence-level claims in `text`."""
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        word = re.search(r"(\S+)$", text[start:m.start() + 1])
        token = word.group(1).rstrip(".!?").lower() if word else ""
        dotted = "." in token and all(len(t) <= 2 for t in token.split("."))  # "L.A."
        if token in _ABBREV or dotted or (len(token) == 1 and token.isalpha()):
            continue
        end = m.end()
        while end > start and text[end - 1].isspace():
            end -= 1
        spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        end = len(text.rstrip())
        spans.append((start + len(text[start:]) - len(text[start:].lstrip()), end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


# Opinion rules: first-person stance, evaluative language, modal advice.
# "best-selling"/"best known" are factual, so they are excluded explicitly.
OPINION_PATTERNS = [
    r"\b(i|we) (think|believe|feel|guess|reckon)\b",
    r"\bin my (opinion|view)\b",
    r"\b(imo|imho)\b",
    r"\b(should|ought to)\b",
    r"\b(best|worst|greatest)\b(?![- ](selling|known|friend|picture|actor|actress|director"
    r"|album|song|film|supporting|original|new|animated|foreign|documentary))",
    r"\b(amazing|awesome|terrible|awful|overrated|underrated|beautiful|ugly|boring"
    r"|disgusting|wonderful|brilliant|stupid)\b",
    r"\b(favou?rite)\b",
]
_OPINION = re.compile("|".join(OPINION_PATTERNS), re.IGNORECASE)

# Written examples of the rule (also used as tests).
OPINION_EXAMPLES = [
    ("I think Paris is the nicest city in Europe.", True),
    ("In my opinion, the film was too long.", True),
    ("Everyone should visit Japan once.", True),
    ("That album is overrated.", True),
    ("Pizza is the best food.", True),
    ("Paris is the capital of France.", False),
    ("Thriller is the best-selling album of all time.", False),
    ("Meryl Streep won the Academy Award for Best Actress.", False),
    ("The Eiffel Tower is 330 metres tall.", False),
    ("Barack Obama was born in 1961.", False),
    ("David Spade appeared in Joe Dirt 2: Beautiful Loser.", False),
    ('"I Feel for You" by Chaka Khan was released in 1984.', False),
]


def is_opinion(claim: str) -> bool:
    """A match counts unless it sits inside a capitalized name or title
    ("Joe Dirt 2: Beautiful Loser", the song "I Feel for You")."""
    for m in _OPINION.finditer(claim):
        words = [w for w in re.findall(r"[A-Za-z]+", m.group(0)) if w not in ("I", "We")]
        if m.start() == 0 or not any(w[0].isupper() for w in words):
            return True
    return False


def _key(text: str) -> str:
    return hashlib.sha1(normalize(text).text.lower().encode()).hexdigest()


def _shingles(text: str, n: int = 3) -> set:
    toks = re.findall(r"\w+", normalize(text).text.lower())
    return {tuple(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))}


def dedup(rows: list[Row], threshold: float = 0.9):
    """Exact-hash then word-3-gram Jaccard de-dup. Returns (kept, exact_drops,
    near_drops); each drop is (dropped_row, kept_row).

    Exact duplicates with the same label have their gold evidence merged into the
    kept row (FEVER repeats a claim once per evidence set). Near duplicates are
    dropped only when the labels agree: FEVER mutation pairs ("named after Queen
    Victoria" / "... Queen Elizabeth") differ by one entity and are distinct claims."""
    seen, kept, exact = {}, [], []
    for r in rows:
        k = _key(r.claim)
        if k in seen:
            first = seen[k]
            if first.label == r.label:
                have = {(e["doc_id"], e["sentence_id"]) for e in first.evidence}
                first.evidence += [e for e in r.evidence
                                   if (e["doc_id"], e["sentence_id"]) not in have]
            exact.append((r, first))
        else:
            seen[k] = replace(r, evidence=list(r.evidence))
            kept.append(seen[k])
    final, near, shingles = [], [], []
    for r in kept:
        sh = _shingles(r.claim)
        match = None
        for other, osh in zip(final, shingles, strict=True):
            inter = len(sh & osh)
            if inter and other.label == r.label and inter / len(sh | osh) >= threshold:
                match = other
                break
        if match:
            near.append((r, match))
        else:
            final.append(r)
            shingles.append(sh)
    return final, exact, near


def split_by_source(rows: list[Row], seed: int = SEED, ratios=RATIOS) -> dict[str, list[Row]]:
    """Assign whole source_id groups to train/dev/test; deterministic for a seed."""
    groups = defaultdict(list)
    for r in rows:
        groups[r.source_id].append(r)
    sources = sorted(groups)
    random.Random(seed).shuffle(sources)
    total, out, acc = len(rows), {"train": [], "dev": [], "test": []}, 0
    bounds = (ratios[0] * total, (ratios[0] + ratios[1]) * total)
    for s in sources:
        name = "train" if acc < bounds[0] else "dev" if acc < bounds[1] else "test"
        out[name] += groups[s]
        acc += len(groups[s])
    return out


def _ex(before: str, after: str) -> str:
    return f"  - before: `{before}`\n  - after: `{after}`"


def main() -> None:
    d = config.setup_caches()
    log = ["# Preprocessing log", "",
           "*Generated by `python -m factcheck.preprocess` (seed 13). Counts are claims "
           "unless stated otherwise.*", ""]
    fever = load(d / "processed" / "fever_rows.jsonl")
    log += ["## 0. Input", f"- FEVER sample rows in: **{len(fever)}** "
            f"({dict(Counter(r.label for r in fever))})", ""]

    # 1. Normalization (offset-preserving). Claims are already clean; the
    # evidence sentences carry PTB escapes, so show one of those.
    changed = [r for r in fever if normalize(r.claim).text != r.claim]
    import json
    pool_ex = None
    with open(d / "processed" / "fever_pool.jsonl") as f:
        for line in f:
            t = json.loads(line)["text"]
            if "-LRB-" in t and len(t) < 160:
                pool_ex = t
                break
    log += ["## 1. Normalization (offset-preserving)",
            "NFKC, PTB escapes (`-LRB-` to `(` and so on), whitespace collapse, and no space "
            "before punctuation. Every output character maps back to a raw span, so offsets "
            "computed on normalized text can be mapped back to `raw_text`. The raw text is "
            "never overwritten.",
            f"- claims in: {len(fever)}, out: {len(fever)} (normalization drops nothing); "
            f"claims whose text changed: {len(changed)}",
            "- example (evidence sentence):", _ex(pool_ex, normalize(pool_ex).text)]
    if changed:
        log += ["- example (claim):", _ex(changed[0].claim, normalize(changed[0].claim).text)]
    log.append("")

    # 2. Claim splitting
    multi = [(r, split_claims(r.claim)) for r in fever]
    multi = [(r, s) for r, s in multi if len(s) > 1]
    demo = "Paris is in France. Dr. Smith lives there. It has 2.1 million people."
    log += ["## 2. Claim splitting",
            "Regex sentence splitter that skips abbreviations and single initials. A FEVER "
            "claim is already one sentence, so the rows are kept whole. Splitting matters "
            "for free text passed to `check()` and to the CLI.",
            f"- claims in: {len(fever)}, rows with more than one sentence: {len(multi)}, "
            f"rows out: {len(fever)}",
            f"- example (free text): `{demo}` gives "
            + ", ".join(f"`{demo[s:e]}`" for s, e in split_claims(demo))]
    if multi:
        r, s = multi[0]
        log.append(f"- FEVER example: `{r.claim}` gives "
                   + ", ".join(f"`{r.claim[a:b]}`" for a, b in s))
    log.append("")

    # 3. Opinion filter
    flagged = [r for r in fever if is_opinion(r.claim)]
    fever = [replace(r, is_opinion=is_opinion(r.claim)) for r in fever]
    log += ["## 3. Opinion filter",
            "Rule-based (first-person stance, evaluative adjectives, `should`), ignoring matches "
            "inside capitalized names and titles. Flagged rows "
            "stay in the data with `is_opinion=true`. `check()` does not verify them and "
            "returns NOT ENOUGH INFO, so any false positive on FEVER counts as an error "
            "in the results.",
            f"- claims in: {len(fever)}, flagged as opinion: {len(flagged)} "
            f"(labels: {dict(Counter(r.label for r in flagged))})"]
    log += [f"  - flagged: `{r.claim}` (gold {r.label})" for r in flagged[:5]]
    log.append("")

    # 4. De-duplication
    kept, exact, near = dedup(fever)
    conflicts = [(a, b) for a, b in exact + near if a.label != b.label]
    log += ["## 4. De-duplication",
            "Exact hash of the normalized, lowercased claim; then word 3-gram Jaccard of at "
            "least 0.9 against the rows already kept, same label only. The first occurrence "
            "is kept, and exact duplicates with the same label pass their gold evidence to "
            "it (FEVER lists a claim once per evidence set). Near-identical claims with "
            "different labels are FEVER mutation pairs and are kept; the split by source "
            "keeps each pair on one side.",
            f"- claims in: {len(fever)}, exact duplicates dropped: {len(exact)}, near "
            f"duplicates dropped: {len(near)}, out: {len(kept)}",
            f"- dropped pairs whose labels disagree: {len(conflicts)}"]
    for a, b in (exact[:1] + near[:2]):
        log += [f"  - dropped `{a.claim}` ({a.label}); kept `{b.claim}` ({b.label})"]
    log.append("")

    # 5. Split by source
    splits = split_by_source(kept)
    log += ["## 5. Split by source",
            f"Whole `source_id` groups (Wikipedia page) are assigned with seed {SEED} and "
            "target ratios 70/15/15. No source appears in two splits.",
            "", "| split | claims | sources | SUPPORTS | REFUTES | NEI |",
            "|---|---|---|---|---|---|"]
    for name, rows in splits.items():
        c = Counter(r.label for r in rows)
        log.append(f"| {name} | {len(rows)} | {len({r.source_id for r in rows})} | "
                   f"{c['SUPPORTS']} | {c['REFUTES']} | {c['NOT ENOUGH INFO']} |")
        save(rows, d / "processed" / f"fever_{name}.jsonl")
    s = [set(r.source_id for r in v) for v in splits.values()]
    log += ["", f"- source overlap train/dev: {len(s[0] & s[1])}, train/test: "
            f"{len(s[0] & s[2])}, dev/test: {len(s[1] & s[2])}", ""]

    # Climate-FEVER: evaluation only, so no split; same normalization and flags.
    climate = load(d / "processed" / "climate_rows.jsonl")
    climate = [replace(r, is_opinion=is_opinion(r.claim)) for r in climate]
    ckept, cexact, cnear = dedup(climate)
    save(ckept, d / "processed" / "climate_test.jsonl")
    log += ["## 6. Climate-FEVER (evaluation only)",
            f"- rows in: {len(climate)} (after dropping DISPUTED at ingest), flagged opinion: "
            f"{sum(r.is_opinion for r in climate)}, exact duplicates: {len(cexact)}, near "
            f"duplicates: {len(cnear)}, out: {len(ckept)}. Everything goes in one "
            "evaluation split; nothing is tuned on it.", ""]

    (config.REPO / "docs" / "preprocessing-log.md").write_text("\n".join(log))
    print("\n".join(log))


if __name__ == "__main__":
    main()
