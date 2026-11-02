import pytest

from factcheck.ingest import Row, ValidationError, validate


def _row(**kw):
    base = dict(id="r1", source_id="s1", raw_text="Intro. The Eiffel Tower is in Berlin.",
                claim="The Eiffel Tower is in Berlin.", claim_start=7, claim_end=37,
                label="REFUTES", wrong_spans=[{"start": 23, "end": 29}])
    base.update(kw)
    return Row(**base)


def test_valid_row_passes():
    validate(_row(), require_refutes_span=True)


def test_claim_offsets_must_round_trip():
    with pytest.raises(ValidationError):
        validate(_row(claim_start=6))


def test_span_outside_claim_rejected():
    with pytest.raises(ValidationError):
        validate(_row(wrong_spans=[{"start": 25, "end": 99}]))


def test_refutes_needs_span_only_when_required():
    validate(_row(wrong_spans=[]))  # FEVER rows: no gold spans
    with pytest.raises(ValidationError):
        validate(_row(wrong_spans=[]), require_refutes_span=True)


def test_bad_label_rejected():
    with pytest.raises(ValidationError):
        validate(_row(label="TRUE", wrong_spans=[]))
