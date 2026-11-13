import csv

import pytest

from factcheck.evaluate import cohen_kappa
from factcheck.ingest import Row
from factcheck.relabel import COLUMNS, build_sheet, parse_label, read_labels, write_sheet

S, R, N = "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"


def test_cohen_kappa_hand_computed():
    # po = 3/4; marginals a: S .5 R .25 N .25, b: S .25 R .5 N .25
    # pe = .125 + .125 + .0625 = .3125; kappa = (.75 - .3125) / .6875
    assert cohen_kappa([S, S, R, N], [S, R, R, N]) == pytest.approx(0.4375 / 0.6875)
    assert cohen_kappa([S, R, N], [S, R, N]) == 1.0
    # chance-level: same marginals, no agreement beyond chance
    assert cohen_kappa([S, R, S, R], [S, S, R, R]) == pytest.approx(0.0)


def _rows():
    def row(lab, i):
        text = f"claim {lab[0]}{i}"  # claim text must not contain the label itself
        return Row(id=f"{lab[0]}{i}", source_id="s", raw_text=text, claim=text,
                   claim_start=0, claim_end=len(text), label=lab)
    return [row(lab, i) for lab in (S, R, N) for i in range(50)]


def test_sheet_is_deterministic_blind_and_stratified():
    a = build_sheet(_rows(), lambda r: "ev", n=30)
    b = build_sheet(list(reversed(_rows())), lambda r: "ev", n=30)
    assert a == b
    assert list(a[0]) == COLUMNS and all(x["label"] == "" for x in a)
    assert not any(lab in str(x.values()) for x in a for lab in (S, R, N))  # no gold leak
    assert sorted(x["id"][0] for x in a).count("S") == 10


def test_second_pass_same_claims_new_order():
    p1 = build_sheet(_rows(), lambda r: "ev", n=30, order_seed=13)
    p2 = build_sheet(_rows(), lambda r: "ev", n=30, order_seed=14)
    assert {x["id"] for x in p1} == {x["id"] for x in p2}
    assert [x["id"] for x in p1] != [x["id"] for x in p2]


def test_parse_and_read_labels(tmp_path):
    assert parse_label(" nei ") == N and parse_label("s") == S
    with pytest.raises(ValueError):
        parse_label("TRUE")
    sheet = build_sheet(_rows(), lambda r: "ev", n=3)
    for x, lab in zip(sheet, ["S", "refutes", "maybe"], strict=True):
        x["label"] = lab
    write_sheet(sheet, tmp_path / "p.csv")
    with pytest.raises(ValueError, match="invalid"):
        read_labels(tmp_path / "p.csv")
    sheet[2]["label"] = "N"
    write_sheet(sheet, tmp_path / "p.csv")
    assert sorted(read_labels(tmp_path / "p.csv").values()) == sorted([S, R, N])
    with open(tmp_path / "p.csv", newline="") as f:
        assert next(csv.reader(f)) == COLUMNS


def test_score_end_to_end_on_filled_copies(tmp_path):
    """Fill copies of the real sheets: pass 1 = FEVER labels, pass 2 = FEVER labels
    with 10 items changed. Needs the data dir; never touches data/relabel/."""
    from factcheck import config
    from factcheck.ingest import load
    from factcheck.relabel import SHEET_DIR, score
    try:
        gold = {r.id: r.label for r in load(config.data_dir() / "processed" / "fever_dev.jsonl")}
    except FileNotFoundError:
        pytest.skip("data dir not configured")
    if not (SHEET_DIR / "pass1.csv").exists():
        pytest.skip("run `python -m factcheck.relabel export` first")
    for p, flip in (("pass1.csv", 0), ("pass2.csv", 10)):
        rows = list(csv.DictReader(open(SHEET_DIR / p, newline="")))
        for i, r in enumerate(rows):
            r["label"] = gold[r["id"]] if i >= flip else {S: R, R: N, N: S}[gold[r["id"]]]
        write_sheet(rows, tmp_path / p)
    md = score("2026-11-15", "2026-11-19", sheet_dir=tmp_path, out=tmp_path / "lq.md")
    assert "| Pass 1 vs FEVER label | 1.000 |" in md
    assert "4 days" in md and "| 100 |" in md
    assert (tmp_path / "lq.md").exists()
