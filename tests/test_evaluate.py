import pytest

from factcheck.evaluate import (accuracy, bootstrap_ci, claim_only_baseline, confusion,
                                fever_hits, macro_f1, majority_baseline, mean, per_class_prf,
                                recall_hits, span_char_f1)

S, R, N = "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"


def test_hand_computed_case_1_perfect():
    g = [S, R, N]
    assert macro_f1(g, g) == 1.0 and accuracy(g, g) == 1.0


def test_hand_computed_case_2():
    # gold S S R N, pred S R R S
    # S: tp1 fp1 fn1 -> P=.5 R=.5 F=.5 ; R: tp1 fp1 fn0 -> P=.5 R=1 F=2/3 ; N: 0
    g, p = [S, S, R, N], [S, R, R, S]
    prf = per_class_prf(g, p)
    assert prf[S] == pytest.approx({"precision": .5, "recall": .5, "f1": .5, "support": 2})
    assert prf[R]["precision"] == .5 and prf[R]["recall"] == 1.0
    assert prf[N]["f1"] == 0.0
    assert macro_f1(g, p) == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    assert confusion(g, p) == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_hand_computed_case_3_all_majority():
    # predicting S everywhere on S S R N: S has P=.5 R=1 F=2/3; others 0
    g = [S, S, R, N]
    p = majority_baseline([S, S, R], len(g))
    assert p == [S] * 4
    assert macro_f1(g, p) == pytest.approx((2 / 3) / 3)
    assert accuracy(g, p) == 0.5


def test_recall_at_k_and_fever_score():
    gold_sets = [{("A", 0)}, {("B", 1)}, set()]
    retrieved = [[("X", 0), ("A", 0)], [("B", 2), ("C", 0)], [("D", 0)]]
    assert recall_hits(gold_sets[:2], retrieved[:2], 1) == [0, 0]
    assert recall_hits(gold_sets[:2], retrieved[:2], 2) == [1, 0]
    # claim 1 right label + evidence; claim 2 right label, no evidence; claim 3 NEI right
    assert fever_hits([S, R, N], [S, R, N], gold_sets, retrieved, k=5) == [1, 0, 1]
    assert fever_hits([S, R, N], [R, R, S], gold_sets, retrieved, k=5) == [0, 0, 0]


def test_span_char_f1_hand_built():
    # claim 1: gold [23,29) (6 chars), pred [20,29) (9 chars) -> overlap 6
    # claim 2: gold [0,4), pred none ; claim 3: no gold, pred [5,7) (2 chars)
    gold = [[[23, 29]], [[0, 4]], []]
    pred = [[[20, 29]], [], [[5, 7]]]
    out = span_char_f1(gold, pred)
    assert out["precision"] == pytest.approx(6 / 11)
    assert out["recall"] == pytest.approx(6 / 10)
    p, r = 6 / 11, 6 / 10
    assert out["f1"] == pytest.approx(2 * p * r / (p + r))
    assert span_char_f1([[[0, 3]]], [[[0, 3]]])["f1"] == 1.0
    # overlapping predicted spans are not double-counted
    assert span_char_f1([[[0, 4]]], [[[0, 3], [2, 4]]])["precision"] == 1.0


def test_bootstrap_constant_and_bounds():
    assert bootstrap_ci(mean, [1, 1, 1, 1]) == (1.0, 1.0, 1.0)
    p, lo, hi = bootstrap_ci(mean, [0, 1] * 50)
    assert p == 0.5 and lo < 0.5 < hi and 0.35 < lo and hi < 0.65
    assert bootstrap_ci(mean, [0, 1] * 50) == (p, lo, hi)  # seeded


def test_claim_only_baseline_learns_cue_words():
    train = ["x is not y"] * 5 + ["x is y"] * 5
    labels = [R] * 5 + [S] * 5
    assert claim_only_baseline(train, labels, ["a is not b", "a is b"]) == [R, S]
