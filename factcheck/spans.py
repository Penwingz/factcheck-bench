"""Rule-based wrong-span candidates.

Flags numbers, dates and capitalized names in the claim that do not appear in
the best evidence sentence, as [start, end) character offsets into the
original (raw) claim.

NOT handled: negation ("is not"), swapped verbs ("directed" vs "produced"),
lowercase common-noun swaps ("football" vs "baseball") and paraphrased
contradictions. Those REFUTES claims get no span.
"""
import re

from .preprocess import normalize

MONTHS = ("january february march april may june july august september october "
          "november december").split()
NUMBER_WORDS = set("""zero one two three four five six seven eight nine ten eleven
twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty
forty fifty sixty seventy eighty ninety hundred thousand million billion first second
third fourth fifth sixth seventh eighth ninth tenth""".split())
# Capitalized words that start sentences or titles but are not names.
CAP_STOP = set("""The A An In On At Of For From By With To And Or But It Its He She
They His Her Their This That These Those There Here Is Was Are Were Be Has Had Have
Not No As If When While After Before During Only Also""".split())
CONNECTORS = {"of", "de", "da", "del", "der", "van", "von", "la", "le", "du", "y", "and", "the"}

_TOKEN = re.compile(r"\d[\d,.]*\d(?:st|nd|rd|th|s)?|\d(?:st|nd|rd|th|s)?|[A-Za-z][\w'À-ɏ-]*")


def _num_key(tok: str) -> str:
    t = re.sub(r"(st|nd|rd|th|s)$", "", tok.lower())
    return t.replace(",", "").rstrip(".")


def _evidence_keys(evidence: str) -> set[str]:
    toks = _TOKEN.findall(normalize(evidence).text)
    keys = {t.lower() for t in toks}
    keys |= {_num_key(t) for t in toks if t[0].isdigit()}
    return keys


def _candidates(text: str) -> list[list[tuple[int, int, str]]]:
    """Groups of (start, end, token) in normalized claim text. A group is a
    capitalized-name run, or a single number / date / number word."""
    toks = [(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(text)]
    groups, run = [], []

    def flush():
        while run and run[-1][2].lower() in CONNECTORS:
            run.pop()
        if run:
            groups.append(list(run))
        run.clear()

    for s, e, t in toks:
        if t[0].isdigit() or t.lower() in NUMBER_WORDS or t.lower() in MONTHS:
            flush()
            groups.append([(s, e, t)])
        elif t[0].isupper() and t not in CAP_STOP:
            run.append((s, e, t))
        elif run and t.lower() in CONNECTORS:
            run.append((s, e, t))
        else:
            flush()
    flush()
    return groups


def find(claim: str, evidence: str) -> list[list[int]]:
    """Wrong-span candidates: [[start, end], ...] offsets into `claim`."""
    norm = normalize(claim)
    keys = _evidence_keys(evidence)
    spans = []
    for group in _candidates(norm.text):
        missing = []
        for s, e, t in group:
            key = _num_key(t) if t[0].isdigit() else t.lower()
            gone = key not in keys and t.lower() not in CONNECTORS
            if gone and missing and missing[-1][1] < s and _only_connectors(
                    norm.text[missing[-1][1]:s]):
                missing[-1] = (missing[-1][0], e)        # merge "Peter" + "Jackson"
            elif gone:
                missing.append((s, e))
        spans += [list(norm.to_raw(s, e)) for s, e in missing]
    return spans


def _only_connectors(gap: str) -> bool:
    return all(w.lower() in CONNECTORS for w in gap.split())
