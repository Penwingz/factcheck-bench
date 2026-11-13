from types import SimpleNamespace

import numpy as np
import pytest

from factcheck.verdict import Thresholds, aggregate, label_order


def test_label_order_read_from_config():
    # the real model card order
    cfg = SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
    assert label_order(cfg) == ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
    # a shuffled order (string keys, as in HF JSON configs) must be followed, not assumed
    cfg = SimpleNamespace(id2label={"0": "NEUTRAL", "1": "CONTRADICTION", "2": "ENTAILMENT"})
    assert label_order(cfg) == ["NOT ENOUGH INFO", "REFUTES", "SUPPORTS"]


def test_label_order_rejects_unknown_labels():
    with pytest.raises(ValueError):
        label_order(SimpleNamespace(id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}))


def test_aggregate_rules():
    th = Thresholds(0.5, 0.5)
    # columns: SUPPORTS, REFUTES, NEI
    assert aggregate(np.array([[0.1, 0.1, 0.8], [0.7, 0.2, 0.1]]), th) == ("SUPPORTS", 1)
    assert aggregate(np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6]]), th) == ("REFUTES", 0)
    assert aggregate(np.array([[0.3, 0.3, 0.4]]), th)[0] == "NOT ENOUGH INFO"
    assert aggregate(np.zeros((0, 3)), th) == ("NOT ENOUGH INFO", -1)
    # both pass: the stronger one wins
    assert aggregate(np.array([[0.6, 0.1, 0.3], [0.1, 0.9, 0.0]]), th) == ("REFUTES", 1)
    # a higher threshold turns a weak SUPPORTS into NEI
    assert aggregate(np.array([[0.6, 0.1, 0.3]]), Thresholds(0.7, 0.7))[0] == "NOT ENOUGH INFO"
