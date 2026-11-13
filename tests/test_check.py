"""check() integration tests. These load the real NLI model (about 280 MB,
cached under the data dir); they are skipped if the data dir is unavailable."""
import pytest

from factcheck import config

try:
    config.data_dir()
except FileNotFoundError:
    pytest.skip("data dir not configured", allow_module_level=True)

from factcheck.run import check, check_text  # noqa: E402
from factcheck.verdict import Thresholds  # noqa: E402

KEYS = {"label", "scores", "evidence_used", "wrong_spans", "is_opinion", "schema_version"}
TH = Thresholds(0.5, 0.5)


def test_check_returns_documented_keys():
    r = check("Paris is the capital of France.", ["Paris is the capital of France."], TH)
    assert set(r) == KEYS
    assert r["schema_version"] == "1.0"
    assert set(r["scores"]) == {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
    assert abs(sum(r["scores"].values()) - 1) < 1e-4
    assert {"text", "source_id"} <= set(r["evidence_used"][0])
    assert r["label"] == "SUPPORTS" and r["wrong_spans"] == []


def test_eiffel_tower_berlin_refuted_with_span():
    claim = "The Eiffel Tower is in Berlin"
    r = check(claim, ["The Eiffel Tower is in Paris."], TH)
    assert r["label"] == "REFUTES"
    assert any(claim[s:e] == "Berlin" or (s <= 23 and e >= 29) for s, e in r["wrong_spans"])
    assert r["evidence_used"][0]["source_id"] == "provided:0"


def test_opinion_not_checked():
    r = check("I think Paris is the nicest city.", ["Paris is in France."], TH)
    assert r["is_opinion"] is True and r["label"] == "NOT ENOUGH INFO"
    assert set(r) == KEYS


def test_check_text_splits_and_keeps_offsets():
    text = "The Eiffel Tower is in Berlin. Paris is the capital of France."
    out = check_text(text, ["The Eiffel Tower is in Paris. Paris is the capital of France."])
    assert [text[o["claim_start"]:o["claim_end"]] for o in out] == [o["claim"] for o in out]
    assert len(out) == 2
