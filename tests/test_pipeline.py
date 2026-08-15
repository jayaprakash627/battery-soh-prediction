"""The charge-to-discharge join, and the evaluation splits.

`pair_charge_to_discharge` is the quietest place a mistake could hide in this
project: pair a charge with the wrong discharge and the model trains on
mislabelled data, scores a bit worse, and looks exactly like a model that needs
more features. So the pairing rules are pinned down here one case at a time.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.build_dataset import METADATA_COLUMNS
from pipeline.train import (
    evaluate_by_cell,
    evaluate_forward_time,
    evaluate_random,
    _standardise,
)
from pipeline.ingest import pair_charge_to_discharge


def op(kind, index):
    return {"type": kind, "op_index": index, "cell_id": "B0001"}


def test_simple_alternating_sequence_pairs_one_to_one():
    ops = [op("charge", 0), op("discharge", 1), op("charge", 2), op("discharge", 3)]
    pairs = pair_charge_to_discharge(ops)
    assert [(p["charge"]["op_index"], p["discharge"]["op_index"]) for p in pairs] \
        == [(0, 1), (2, 3)]


def test_impedance_between_the_two_does_not_break_the_pair():
    """The dataset interleaves EIS sweeps routinely; they are measurements, not cycling."""
    ops = [op("charge", 0), op("impedance", 1), op("discharge", 2)]
    pairs = pair_charge_to_discharge(ops)
    assert len(pairs) == 1
    assert pairs[0]["discharge"]["op_index"] == 2


def test_two_charges_in_a_row_keeps_only_the_second():
    """The first charge can never be labelled, so it is dropped rather than
    paired with a discharge that happened after another full charge."""
    ops = [op("charge", 0), op("charge", 1), op("discharge", 2)]
    pairs = pair_charge_to_discharge(ops)
    assert len(pairs) == 1
    assert pairs[0]["charge"]["op_index"] == 1


def test_a_trailing_charge_with_no_discharge_is_dropped():
    ops = [op("charge", 0), op("discharge", 1), op("charge", 2)]
    assert len(pair_charge_to_discharge(ops)) == 1


def test_a_leading_discharge_with_no_charge_is_ignored():
    ops = [op("discharge", 0), op("charge", 1), op("discharge", 2)]
    pairs = pair_charge_to_discharge(ops)
    assert len(pairs) == 1
    assert pairs[0]["charge"]["op_index"] == 1


def test_a_discharge_is_never_used_twice():
    ops = [op("charge", 0), op("discharge", 1), op("discharge", 2)]
    pairs = pair_charge_to_discharge(ops)
    assert len(pairs) == 1


def test_empty_input_is_not_an_error():
    assert pair_charge_to_discharge([]) == []


# --- evaluation -------------------------------------------------------------

def test_metadata_columns_are_never_offered_as_features():
    """The guard against an identifier becoming a feature by accident."""
    from app.features import FEATURE_NAMES
    assert not (set(METADATA_COLUMNS) & set(FEATURE_NAMES))


def test_standardiser_is_fitted_on_the_training_fold_only():
    """Leakage check: the returned mean must be the *training* mean.

    If someone later "simplifies" this by scaling the whole matrix first, every
    score improves slightly and nothing looks wrong. This is the tripwire.
    """
    train = np.array([[0.0], [2.0]])
    test = np.array([[100.0]])
    train_s, test_s, mean, scale = _standardise(train, test)
    assert mean[0] == pytest.approx(1.0)      # mean of train only, not of both
    assert train_s.mean() == pytest.approx(0.0)
    assert test_s[0, 0] > 50                  # test is far out, as it should be


def test_constant_feature_does_not_produce_nan():
    train = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    scaled, _, _ = _standardise(train)
    assert np.all(np.isfinite(scaled))


def _toy_dataset(n_cells=3, per_cell=25):
    """A dataset where SoH is a clean linear function of one feature."""
    rows, X, y, groups = [], [], [], []
    rng = np.random.default_rng(0)
    for c in range(n_cells):
        for k in range(per_cell):
            soh = 100 - 20 * (k / per_cell)
            feats = [soh * 10 + rng.normal(0, 1)] + list(rng.normal(0, 1, 7))
            X.append(feats)
            y.append(soh)
            groups.append(f"S{c}")
            rows.append({"cycle_number": k + 1})
    return np.array(X), np.array(y), np.array(groups), rows


def test_by_cell_evaluation_holds_out_whole_cells():
    from sklearn.linear_model import Ridge
    X, y, groups, _ = _toy_dataset()
    pooled, per_cell, preds = evaluate_by_cell(lambda: Ridge(alpha=1.0), X, y, groups)
    assert set(per_cell) == {"S0", "S1", "S2"}
    assert sum(m["n"] for m in per_cell.values()) == len(y)
    assert pooled["mae_pct"] < 1.0  # the signal is clean, so it should do well


def test_forward_time_split_trains_on_early_life_only():
    from sklearn.linear_model import Ridge
    X, y, groups, rows = _toy_dataset()
    out = evaluate_forward_time(lambda: Ridge(alpha=1.0), X, y, groups, rows,
                                train_frac=0.6)
    assert out["train_n"] < len(y)
    assert out["n"] + out["train_n"] == len(y)


def test_random_split_covers_every_row_exactly_once():
    from sklearn.linear_model import Ridge
    X, y, _, _ = _toy_dataset()
    m = evaluate_random(lambda: Ridge(alpha=1.0), X, y, folds=5)
    assert m["n"] == len(y)
