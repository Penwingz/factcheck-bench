import pytest

from factcheck.ingest import Row
from factcheck.preprocess import (OPINION_EXAMPLES, dedup, is_opinion, normalize,
                                  split_by_source, split_claims)


def test_offsets_round_trip_after_normalization():
    raw = "The  Eiffel Tower -LRB- Paris -RRB- is in  Berlin ."
    n = normalize(raw)
    assert n.text == "The Eiffel Tower (Paris) is in Berlin."
    for word in ("Eiffel", "Paris", "Berlin"):
        s = n.text.index(word)
        rs, re_ = n.to_raw(s, s + len(word))
        assert raw[rs:re_] == word
    # a normalized char produced by an escape maps to the whole escape
    p = n.text.index("(")
    assert raw[slice(*n.to_raw(p, p + 1))] == "-LRB-"


def test_normalize_identity_on_clean_text():
    raw = "Barack Obama was born in 1961."
    n = normalize(raw)
    assert n.text == raw
    assert n.to_raw(0, len(raw)) == (0, len(raw))


def test_nfkc_ligature_maps_to_one_raw_char():
    n = normalize("eﬃcient")  # "ffi" ligature
    assert n.text == "efficient"
    assert n.to_raw(1, 4) == (1, 2)


def test_split_claims_offsets():
    text = "Paris is in France. Dr. Smith lives there. J. R. R. Tolkien wrote it! L.A. Reid sang."
    spans = split_claims(text)
    assert [text[s:e] for s, e in spans] == [
        "Paris is in France.", "Dr. Smith lives there.", "J. R. R. Tolkien wrote it!",
        "L.A. Reid sang."]


def test_split_single_sentence():
    assert split_claims("The Eiffel Tower is in Berlin") == [(0, 29)]


@pytest.mark.parametrize("text,expected", OPINION_EXAMPLES)
def test_opinion_examples(text, expected):
    assert is_opinion(text) is expected


def _rows(n, sources):
    return [Row(id=f"r{i}", source_id=sources[i % len(sources)], raw_text=f"claim {i}",
                claim=f"claim {i}", claim_start=0, claim_end=len(f"claim {i}"),
                label="SUPPORTS") for i in range(n)]


def test_split_is_deterministic():
    rows = _rows(200, [f"s{i}" for i in range(37)])
    a = split_by_source(rows, seed=13)
    b = split_by_source(list(reversed(rows)), seed=13)
    for k in a:
        assert sorted(r.id for r in a[k]) == sorted(r.id for r in b[k])
    c = split_by_source(rows, seed=14)
    assert any(sorted(r.id for r in a[k]) != sorted(r.id for r in c[k]) for k in a)


def test_no_source_overlap():
    rows = _rows(500, [f"s{i}" for i in range(80)])
    sp = split_by_source(rows, seed=13)
    sets = [{r.source_id for r in v} for v in sp.values()]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(v) for v in sp.values()) == 500
    assert all(sp.values())


def test_dedup_exact_and_near():
    def r(i, text, label="SUPPORTS"):
        return Row(id=str(i), source_id="s", raw_text=text, claim=text, claim_start=0,
                   claim_end=len(text), label=label)
    rows = [r(1, "Paris is the capital city of France ."),
            r(2, "paris is the capital city of france."),           # exact after normalize
            r(3, "Paris is the capital city of the French Republic in Europe."),
            r(4, "Paris is the capital city of the French Republic in Europe today."),  # near
            r(5, "Paris is the capital city of the French Republic in Asia today.",
              "REFUTES")]  # same words, different label: a mutation pair, kept
    kept, exact, near = dedup(rows)
    assert [x.id for x in kept] == ["1", "3", "5"]
    assert [a.id for a, _ in exact] == ["2"] and [a.id for a, _ in near] == ["4"]


def test_exact_duplicates_merge_evidence():
    def r(i, ev):
        t = "Paris is in France."
        return Row(id=str(i), source_id="s", raw_text=t, claim=t, claim_start=0,
                   claim_end=len(t), label="SUPPORTS",
                   evidence=[{"doc_id": "Paris", "sentence_id": ev, "text": "x"}])
    kept, exact, _ = dedup([r(1, 0), r(2, 3)])
    assert len(kept) == 1 and [e["sentence_id"] for e in kept[0].evidence] == [0, 3]
