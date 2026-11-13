from factcheck.config import REPO
from factcheck.ingest import load
from factcheck.spans import find


def _texts(claim, evidence):
    return [claim[s:e] for s, e in find(claim, evidence)]


def test_entity_number_and_date_swaps():
    assert _texts("The Eiffel Tower is in Berlin", "The Eiffel Tower is in Paris.") == ["Berlin"]
    assert _texts("Barack Obama was born in 1962.", "Barack Obama was born in 1961.") == ["1962"]
    assert _texts("It has 145,170 people.", "It has 145170 people.") == []
    assert _texts("Hamlet was written by Charles Dickens.",
                  "Hamlet is a tragedy by William Shakespeare.") == ["Charles Dickens"]


def test_offsets_are_into_the_raw_claim():
    claim = "The  Eiffel   Tower is in Berlin"   # irregular whitespace
    (s, e), = find(claim, "The Eiffel Tower is in Paris.")
    assert claim[s:e] == "Berlin"


def test_known_misses_are_documented_not_hidden():
    # negation and common-noun swaps are not handled
    assert find("Einstein did not win the Nobel Prize.", "Einstein won the Nobel Prize.") == []
    assert find("It is for baseball clubs.", "It is for football clubs.") == []


def test_handbuilt_set_validates():
    rows = load(REPO / "data" / "span_gold_handbuilt.jsonl", require_refutes_span=True)
    assert len(rows) == 20 and all(r.annotator == "claude-handbuilt" for r in rows)
