"""The serving layer: does it add up, and does it admit when it is guessing?"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.features import FEATURE_NAMES, extract
from app.model import END_OF_LIFE_SOH_PCT, Model
from tests.synthetic import charge_curve


def test_contributions_account_for_the_prediction_exactly(model):
    """Not "roughly" — exactly.

    The UI tells the user these bars *are* the answer, broken down. On a linear
    model over standardised features that is arithmetically true, and this test
    is what stops a future refactor from quietly making it merely approximate.
    """
    pred = model.predict(extract(**charge_curve(capacity_ah=1.7)))
    total = pred.baseline_pct + sum(c.effect_pct for c in pred.contributions)
    assert total == pytest.approx(pred.soh_pct, abs=0.02)


def test_contributions_are_ordered_by_influence(model):
    pred = model.predict(extract(**charge_curve(capacity_ah=1.7)))
    sizes = [abs(c.effect_pct) for c in pred.contributions]
    assert sizes == sorted(sizes, reverse=True)


def test_every_feature_is_reported(model):
    pred = model.predict(extract(**charge_curve()))
    assert {c.feature for c in pred.contributions} == set(FEATURE_NAMES)
    assert all(c.physics for c in pred.contributions)


def test_lower_capacity_predicts_lower_soh(model):
    healthy = model.predict(extract(**charge_curve(capacity_ah=2.0)))
    degraded = model.predict(extract(**charge_curve(capacity_ah=1.5)))
    assert degraded.soh_pct < healthy.soh_pct


def test_interval_brackets_the_estimate(model):
    pred = model.predict(extract(**charge_curve(capacity_ah=1.8)))
    assert pred.interval_low_pct < pred.soh_pct < pred.interval_high_pct


def test_wider_confidence_gives_a_wider_interval(model):
    f = extract(**charge_curve(capacity_ah=1.8))
    narrow = model.predict(f, confidence=0.50)
    wide = model.predict(f, confidence=0.95)
    assert (wide.interval_high_pct - wide.interval_low_pct) >= \
           (narrow.interval_high_pct - narrow.interval_low_pct)


def test_unsupported_confidence_is_rejected(model):
    with pytest.raises(ValueError, match="confidence"):
        model.predict(extract(**charge_curve()), confidence=0.8)


def test_end_of_life_uses_the_declared_threshold(model):
    """The flag must follow the constant, not a number typed twice."""
    pred = model.predict(extract(**charge_curve(capacity_ah=1.8)))
    assert pred.end_of_life == (pred.soh_pct < END_OF_LIFE_SOH_PCT)


def test_far_out_input_is_flagged_as_extrapolation(model):
    """A cell far outside the training distribution must be labelled, not answered silently."""
    normal = model.predict(extract(**charge_curve(capacity_ah=1.8)))
    assert not normal.extrapolating

    weird = extract(**charge_curve(capacity_ah=1.8))
    weird["temp_max_c"] = 300.0  # a pack this hot is on fire, not being measured
    flagged = model.predict(weird)
    assert flagged.extrapolating
    assert any("standard deviations" in n for n in flagged.notes)


def test_model_refuses_an_artifact_whose_features_do_not_match(model_artifact):
    """A stale model.json against newer feature code must fail loudly.

    This is the dangerous case: reordered coefficients still multiply cleanly
    and produce a plausible number. Nothing would crash and every answer would
    be wrong.
    """
    artifact = json.loads(model_artifact.read_text())
    artifact["feature_names"] = list(reversed(artifact["feature_names"]))
    with pytest.raises(ValueError, match="different features"):
        Model(artifact)


def test_model_refuses_an_unknown_kind(model_artifact):
    artifact = json.loads(model_artifact.read_text())
    artifact["kind"] = "neural_net"
    with pytest.raises(ValueError, match="unsupported model kind"):
        Model(artifact)


def test_artifact_is_plain_json_not_a_pickle(model_artifact):
    """Serving must never involve unpickling. Reading it as text should just work."""
    parsed = json.loads(model_artifact.read_text())
    assert parsed["kind"] == "ridge"
    assert len(parsed["coefficients"]) == len(FEATURE_NAMES)
    assert set(parsed["standardiser"]) == {"mean", "scale"}


def test_prediction_matches_a_hand_computed_dot_product(model):
    """Independent arithmetic, in case predict() ever grows a bug it agrees with."""
    features = extract(**charge_curve(capacity_ah=1.65))
    x = np.array([features[n] for n in FEATURE_NAMES])
    expected = model.intercept + float(
        np.dot(model.coef, (x - model.mean) / model.scale))
    assert model.predict(features).soh_pct == pytest.approx(expected, abs=0.01)
