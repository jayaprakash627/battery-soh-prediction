"""Does the feature extraction read the physics it claims to read?

Two kinds of test here. The first kind turns one dial on the synthetic cell and
asserts the feature moves the way FEATURES says it should — if the docstring
claims "longer = healthier", a test proves the sign. The second kind checks that
unreadable curves are refused rather than answered, because a wrong number
returned confidently is the failure mode this project is built to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.features import (
    FEATURE_NAMES,
    WINDOW_HIGH_V,
    WINDOW_LOW_V,
    UnusableCharge,
    extract,
    to_vector,
)
from tests.synthetic import charge_curve


def test_extract_returns_every_declared_feature():
    f = extract(**charge_curve())
    assert set(f) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in f.values())


def test_no_history_features_leak_in():
    """The model must not be able to see cycle count or capacity.

    This is the project's central claim, so it is asserted rather than trusted:
    if someone later adds a convenient `cycle_number` feature, this fails.
    """
    forbidden = {"cycle", "cycle_number", "capacity", "capacity_ah",
                 "soh", "soh_pct", "cell_id", "age"}
    assert not (set(FEATURE_NAMES) & forbidden)


@pytest.mark.parametrize("feature,expected_sign", [
    # As capacity falls, these fall too.
    ("window_charge_ah", -1),
    ("ic_peak_height", -1),
    # ...and this rises: less capacity means voltage climbs faster.
    ("voltage_slope_v_per_s", +1),
])
def test_capacity_loss_moves_features_the_declared_way(feature, expected_sign):
    healthy = extract(**charge_curve(capacity_ah=2.0, resistance_ohm=0.05))
    degraded = extract(**charge_curve(capacity_ah=1.4, resistance_ohm=0.05))
    change = degraded[feature] - healthy[feature]
    assert np.sign(change) == expected_sign, (
        f"{feature}: healthy={healthy[feature]:.5g} degraded={degraded[feature]:.5g}"
    )


def test_window_charge_is_proportional_to_capacity():
    """Across a fixed voltage window, charge accepted should track capacity.

    Checked as a ratio rather than a direction because this is the strongest
    feature in the model, and "it went down" would still pass if it went down by
    a thousandth of what the physics requires.
    """
    a = extract(**charge_curve(capacity_ah=2.0))["window_charge_ah"]
    b = extract(**charge_curve(capacity_ah=1.0))["window_charge_ah"]
    assert 0.45 < b / a < 0.60


def test_rising_resistance_raises_temperature():
    cool = extract(**charge_curve(capacity_ah=1.8, resistance_ohm=0.03))
    hot = extract(**charge_curve(capacity_ah=1.8, resistance_ohm=0.09))
    assert hot["temp_rise_c"] > cool["temp_rise_c"]


def test_negative_current_convention_is_normalised():
    """A charger that logs charging current as negative must not invert the answer."""
    curve = charge_curve(capacity_ah=1.8)
    flipped = dict(curve, current_a=[-i for i in curve["current_a"]])
    assert extract(**curve)["window_charge_ah"] == pytest.approx(
        extract(**flipped)["window_charge_ah"], rel=1e-9)


def test_noise_does_not_move_the_answer_much():
    clean = extract(**charge_curve(capacity_ah=1.8))
    noisy = extract(**charge_curve(capacity_ah=1.8, seed=7))
    assert clean["window_charge_ah"] == pytest.approx(
        noisy["window_charge_ah"], abs=0.02)


def test_coarse_logging_still_works():
    """A charger sampling once a minute instead of every 5 s."""
    f = extract(**charge_curve(capacity_ah=1.8, sample_seconds=60))
    assert f["window_charge_ah"] > 0


# --- the refusals -----------------------------------------------------------

def test_part_charged_pack_is_refused():
    """Starting above the window means the window is truncated, not shorter."""
    with pytest.raises(UnusableCharge, match="part-charged"):
        extract(**charge_curve(capacity_ah=1.8, start_soc=0.85))


def test_charge_that_never_reaches_the_window_is_refused():
    curve = charge_curve(capacity_ah=1.8)
    n = len(curve["time_s"])
    cut = next(i for i in range(n) if curve["voltage_v"][i] > WINDOW_LOW_V + 0.02)
    truncated = {k: v[:cut] for k, v in curve.items()}
    with pytest.raises(UnusableCharge):
        extract(**truncated)


def test_mismatched_channel_lengths_are_refused():
    curve = charge_curve()
    curve["voltage_v"] = curve["voltage_v"][:-3]
    with pytest.raises(UnusableCharge, match="channel lengths differ"):
        extract(**curve)


def test_too_few_samples_is_refused():
    curve = charge_curve()
    with pytest.raises(UnusableCharge, match="at least 20"):
        extract(**{k: v[:10] for k, v in curve.items()})


def test_nan_is_refused():
    curve = charge_curve()
    curve["voltage_v"][50] = float("nan")
    with pytest.raises(UnusableCharge, match="NaN"):
        extract(**curve)


def test_unsorted_time_is_refused():
    curve = charge_curve()
    curve["time_s"][10], curve["time_s"][40] = curve["time_s"][40], curve["time_s"][10]
    with pytest.raises(UnusableCharge, match="increasing order"):
        extract(**curve)


def test_to_vector_is_ordered_by_declaration_not_insertion():
    f = extract(**charge_curve())
    shuffled = dict(reversed(list(f.items())))
    assert np.array_equal(to_vector(f), to_vector(shuffled))


def test_to_vector_rejects_a_missing_feature():
    f = extract(**charge_curve())
    del f[FEATURE_NAMES[0]]
    with pytest.raises(UnusableCharge, match="missing features"):
        to_vector(f)


def test_window_bounds_are_inside_the_constant_current_phase():
    """Guards the assumption the timing features rest on."""
    curve = charge_curve(capacity_ah=1.8)
    v = np.array(curve["voltage_v"])
    i = np.array(curve["current_a"])
    in_window = (v >= WINDOW_LOW_V) & (v <= WINDOW_HIGH_V)
    # Current should still be at its constant-current value throughout.
    assert np.ptp(i[in_window]) < 1e-6
