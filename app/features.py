"""Turn one charge curve into the numbers the model actually reads.

This module is imported by the training pipeline *and* by the API. That is
deliberate and it is the most important structural decision in the project.
The classic way to ruin a deployed model is to extract features one way in the
notebook and a subtly different way in the service — a different smoothing
window, degrees instead of seconds, a filter applied in one place and not the
other. The model keeps returning confident numbers and they are all slightly
wrong, and nothing crashes. Sharing one function makes that class of bug
impossible rather than unlikely.

The idea
--------
Measuring a battery's true capacity means fully discharging it, which takes
hours and takes the vehicle off the road. Nobody does this to a customer's
e-bike. But every pack gets charged, and a charger already measures voltage,
current and temperature.

So the question this project asks is not "can a model fit a capacity curve" —
that is trivially yes — but the one a fleet operator actually has:

    Given a few minutes of an ordinary charge, and knowing nothing about this
    pack's history, how much capacity does it have left?

Everything below follows from that. A feature is allowed here only if a charger
could compute it from a *single* charge event, with no memory of the cell.

What is deliberately NOT a feature
----------------------------------
  cycle number        The strongest predictor in the dataset and useless in the
                      field. It leaks the answer: capacity falls with cycle
                      count by construction, so a model given cycle number
                      learns the experiment's schedule, not the battery's
                      physics. A second-hand pack does not arrive with its cycle
                      count written on it.
  previous capacity   Same problem, worse. Capacity is what we are predicting.
  cell identity       Eight cells is not enough to learn per-cell offsets that
                      would generalise; it is exactly enough to memorise them.

The physics behind each feature is spelled out in FEATURES below, because a
number nobody can explain is a number nobody will act on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The voltage window the features are measured over. Both bounds sit inside the
# constant-current phase of a standard Li-ion CC-CV charge, which matters: in CC
# the charger holds current fixed and lets voltage climb, so the *time* taken to
# cross the window is a clean read on how much charge the cell can still accept.
# In the constant-voltage phase that logic inverts and the two phases would
# average into mush.
#
# 3.90 V is above the flat plateau where voltage barely moves (tiny voltage
# errors would become huge time errors); 4.15 V stops short of the 4.2 V
# changeover so a cell that starts tapering early still produces a full window.
WINDOW_LOW_V = 3.90
WINDOW_HIGH_V = 4.15

# A charge is only usable if it starts *well* below the window. Not merely below
# it: the cell has to enter the window with the constant-current phase already
# established, or the traversal is truncated and the timing reads short.
#
# This margin was originally 0.01 V, which satisfied the letter of the rule and
# missed its point. Three charges in the NASA set begin at 3.865–3.873 V — some
# 30 mV under the window — and they cross it roughly six times faster than the
# next cycle of the same cell at the same measured capacity. The cell had not
# aged between those two charges; it simply started most of the way full. Left
# in, they were the model's three worst predictions by a wide margin: one cell's
# healthiest cycle, 101.8% SoH, was read as 74.9%.
#
# 0.15 V below the window start is about 5% state of charge — enough run-up for
# the current to settle, and roughly five times the ~30 mV that caused the
# failure. It costs 3 of 902 cycles, and moved leave-one-cell-out MAE from 1.74
# to 1.63 points and the early-life-to-late-life split from 3.11 to 1.70.
MIN_RUNUP_V = 0.15

# The constant-voltage phase is considered finished when current falls below
# this. NASA's protocol cuts off at 20 mA; measuring to a slightly higher
# threshold makes the feature robust to the last few noisy samples.
CV_END_CURRENT_A = 0.05

# Median filter width for the voltage channel before it is used monotonically.
# Wide enough to kill single-sample spikes, narrow enough not to smear the
# curvature that the dQ/dV features read. Odd so the window is symmetric.
SMOOTH_SAMPLES = 5


@dataclass(frozen=True)
class Feature:
    """One model input, with the reason it belongs in the model."""

    name: str
    unit: str
    physics: str      # why this number moves as a cell ages
    direction: str    # what a *rising* value means for health


FEATURES: tuple[Feature, ...] = (
    Feature(
        name="window_charge_ah",
        unit="Ah",
        physics=(
            "How much charge the battery takes in while its voltage climbs from "
            "3.90 V to 4.15 V. An older battery holds less, so it needs less "
            "charge to cross that range."
        ),
        direction="higher = healthier",
    ),
    Feature(
        name="ic_peak_height",
        unit="Ah/V",
        physics=(
            "The biggest jump in stored charge for a small step in voltage. It "
            "shows how much of the battery is still doing work. As a battery "
            "wears out, this gets smaller."
        ),
        direction="higher = healthier",
    ),
    Feature(
        name="ic_peak_voltage",
        unit="V",
        physics=(
            "The voltage at which that jump happens. An old battery has more "
            "internal resistance, which pushes its voltage up. So this creeps "
            "higher with age."
        ),
        direction="lower = healthier",
    ),
    Feature(
        name="voltage_slope_v_per_s",
        unit="V/s",
        physics=(
            "How fast the voltage climbs through the range. A battery that "
            "holds less charge fills up sooner, so its voltage rises faster."
        ),
        direction="lower = healthier",
    ),
    Feature(
        name="cv_phase_seconds",
        unit="s",
        physics=(
            "How long the charger spends topping up at the end, after the "
            "voltage has hit its limit. A worn battery reaches that limit early "
            "while still part empty, then takes the rest slowly."
        ),
        direction="shorter = healthier",
    ),
    Feature(
        name="temp_rise_c",
        unit="°C",
        physics=(
            "How much the battery warms up while crossing the range. Resistance "
            "grows as a battery ages, and more resistance means more heat for "
            "the same current."
        ),
        direction="smaller = healthier",
    ),
    Feature(
        name="temp_max_c",
        unit="°C",
        physics=(
            "The hottest the battery gets during the whole charge. This is "
            "partly the battery and partly the room, which is why the warm-up "
            "above is kept separate."
        ),
        direction="lower = healthier",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)

# One feature was measured, found to be an exact duplicate, and removed.
#
# The first version of this model also carried `cc_window_seconds`: the time
# taken to cross the same voltage window. On this data it correlated with
# `window_charge_ah` at r = +1.00 — not "highly", exactly. That is not a
# coincidence to be regularised away, it is arithmetic: inside the
# constant-current phase the current is fixed, so charge is current times time
# and the two columns are the same measurement in different units.
#
# `window_charge_ah` is the one kept, and not because it scored better (0.05
# points of MAE apart, which is noise). It survives contact with a real
# charger: integrated charge stays meaningful when the current wobbles, where
# a voltage-crossing time silently becomes a measurement of the charger.
#
# Features that remain correlated with each other are declared below rather
# than pruned, because dropping them costs real accuracy — see
# CORRELATION_WARNING.
COLLINEARITY_THRESHOLD = 0.90

# Shown wherever per-feature contributions are displayed. This project's whole
# premise is explaining its own numbers, so the limit of that explanation has to
# be stated in the same breath, not buried:
#
# `window_charge_ah`, `ic_peak_height` and `voltage_slope_v_per_s` are all
# reading the same underlying thing — how much charge this cell accepts — and
# they correlate with each other above 0.9. When inputs are that entangled, the
# split of credit *between* them is not identified: the model can shift weight
# from one to another, and even flip a sign, with almost no change to the
# prediction. `voltage_slope_v_per_s` does exactly that here, taking a positive
# coefficient although its own correlation with SoH is negative.
#
# The prediction is not harmed by this and neither is the total. What it means
# is that a single bar is an accounting entry, not a physical claim: the group
# is interpretable, the individual split is not. Regularising until every sign
# agreed with its physics was tried — it takes alpha ≈ 1000, and by-cell MAE
# goes from 1.7 to 4.0. Buying a tidier story with twice the error would be the
# wrong trade, and pretending the story was tidy would be worse.
CORRELATION_WARNING = (
    "Three of these bars — charge taken in, the charge jump, and how fast voltage "
    "rises — all track the same thing, so read them as one group. The group total "
    "is solid. How the model splits credit between those three is not, and one of "
    "them can even point the wrong way."
)


class UnusableCharge(ValueError):
    """The curve cannot produce an honest feature vector.

    Raised rather than returning NaN or a zero-filled vector. A model handed
    zeros for a missing window will answer anyway, confidently and wrongly; a
    fleet tool must say "I cannot read this charge" instead. The API turns this
    into a 422 with the reason attached.
    """


def _median_filter(x: np.ndarray, width: int) -> np.ndarray:
    """Small median filter, edge-padded. Written out rather than pulled from
    scipy.signal so the deployed service does not need scipy — it needs numpy
    and nothing else."""
    if width <= 1 or x.size < width:
        return x.copy()
    pad = width // 2
    padded = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=1)


def _first_crossing(voltage: np.ndarray, time: np.ndarray, level: float) -> float:
    """Time at which voltage first reaches `level`, linearly interpolated.

    Interpolated rather than snapped to the nearest sample because NASA logs
    roughly every 10 s, and on the steep part of the curve a whole sample is
    worth several seconds of the very quantity being measured. Snapping would
    quantise the strongest feature in the model into ~10 s steps.
    """
    above = np.flatnonzero(voltage >= level)
    if above.size == 0:
        raise UnusableCharge(
            f"This charge never reaches {level:.2f} V. It stops at "
            f"{voltage.max():.3f} V, so the measuring range is incomplete."
        )
    i = int(above[0])
    if i == 0:
        return float(time[0])
    v0, v1 = voltage[i - 1], voltage[i]
    t0, t1 = time[i - 1], time[i]
    if v1 == v0:
        return float(t1)
    return float(t0 + (level - v0) * (t1 - t0) / (v1 - v0))


def _incremental_capacity(voltage: np.ndarray, charge_ah: np.ndarray,
                          low: float, high: float) -> tuple[float, float]:
    """Peak height and location of dQ/dV within [low, high].

    dQ/dV is differentiated on a fixed voltage grid rather than sample-to-sample.
    Raw differences of Q against V blow up wherever two consecutive samples sit
    at nearly the same voltage — dividing by a near-zero ΔV produces spikes that
    have nothing to do with the electrode and everything to do with the sampling
    clock. Resampling first makes the derivative a property of the curve.
    """
    grid = np.linspace(low, high, 60)

    # np.interp needs an increasing x. Voltage in CC is monotonic in principle
    # and jitters in practice, so enforce it rather than assume it.
    order = np.argsort(voltage)
    v_sorted = voltage[order]
    q_sorted = charge_ah[order]

    q_on_grid = np.interp(grid, v_sorted, q_sorted)
    dqdv = np.gradient(q_on_grid, grid)

    peak = int(np.argmax(dqdv))
    return float(dqdv[peak]), float(grid[peak])


def extract(
    *,
    time_s,
    voltage_v,
    current_a,
    temperature_c,
) -> dict[str, float]:
    """Compute the model's inputs from one charge curve.

    Takes the four channels a charger already measures. Raises UnusableCharge
    with a human-readable reason if the curve cannot support an honest answer.
    """
    t = np.asarray(time_s, dtype=float).reshape(-1)
    v = np.asarray(voltage_v, dtype=float).reshape(-1)
    i = np.asarray(current_a, dtype=float).reshape(-1)
    temp = np.asarray(temperature_c, dtype=float).reshape(-1)

    if not (t.size == v.size == i.size == temp.size):
        raise UnusableCharge(
            f"The four readings have different lengths: time={t.size}, "
            f"voltage={v.size}, current={i.size}, temperature={temp.size}. "
            f"They must all match."
        )
    if t.size < 20:
        raise UnusableCharge(
            f"Only {t.size} readings in this charge. At least 20 are needed."
        )
    if not np.all(np.isfinite(t) & np.isfinite(v) & np.isfinite(i) & np.isfinite(temp)):
        raise UnusableCharge(
            "Some readings are missing or not a number."
        )
    if np.any(np.diff(t) < 0):
        raise UnusableCharge(
            "The timestamps go backwards, so the readings are out of order."
        )

    # Charging current is signed positive here. NASA logs it positive on charge
    # and negative on discharge; a charger that reports the other convention
    # would otherwise silently integrate to a negative charge.
    if np.median(i) < 0:
        i = -i

    v_smooth = _median_filter(v, SMOOTH_SAMPLES)

    if v_smooth[0] > WINDOW_LOW_V - MIN_RUNUP_V:
        raise UnusableCharge(
            f"This charge starts at {v_smooth[0]:.3f} V. It needs to start below "
            f"{WINDOW_LOW_V - MIN_RUNUP_V:.2f} V. The battery was already part "
            f"charged, so it races through the measuring range and every timing "
            f"reads too low."
        )

    t_low = _first_crossing(v_smooth, t, WINDOW_LOW_V)
    t_high = _first_crossing(v_smooth, t, WINDOW_HIGH_V)
    duration = t_high - t_low
    if duration <= 0:
        raise UnusableCharge("window end is not after window start")

    # Cumulative charge in amp-hours, trapezoidal over the whole curve.
    charge_ah = np.concatenate(([0.0], np.cumsum(
        np.diff(t) * (i[1:] + i[:-1]) / 2.0))) / 3600.0

    q_low = float(np.interp(t_low, t, charge_ah))
    q_high = float(np.interp(t_high, t, charge_ah))

    in_window = (t >= t_low) & (t <= t_high)
    if in_window.sum() < 5:
        raise UnusableCharge(
            f"Only {int(in_window.sum())} readings fall inside the measuring "
            f"range. The charger logged too rarely to read this charge."
        )

    slope = float(np.polyfit(t[in_window], v_smooth[in_window], 1)[0])

    peak_height, peak_voltage = _incremental_capacity(
        v_smooth[in_window], charge_ah[in_window], WINDOW_LOW_V, WINDOW_HIGH_V
    )

    # CV phase: from the last time voltage is still climbing to the limit, until
    # current has decayed below the cut-off. If the log ends before the current
    # decays (the operator stopped early), the phase is measured to the end of
    # the log and is a lower bound — recorded as-is rather than guessed at.
    v_limit = float(np.max(v_smooth))
    at_limit = np.flatnonzero(v_smooth >= v_limit - 0.02)
    t_cv_start = float(t[at_limit[0]]) if at_limit.size else float(t[-1])
    decayed = np.flatnonzero((t > t_cv_start) & (i < CV_END_CURRENT_A))
    t_cv_end = float(t[decayed[0]]) if decayed.size else float(t[-1])
    cv_seconds = max(0.0, t_cv_end - t_cv_start)

    temp_low = float(np.interp(t_low, t, temp))
    temp_high = float(np.interp(t_high, t, temp))

    return {
        "window_charge_ah": q_high - q_low,
        "ic_peak_height": peak_height,
        "ic_peak_voltage": peak_voltage,
        "voltage_slope_v_per_s": slope,
        "cv_phase_seconds": cv_seconds,
        "temp_rise_c": temp_high - temp_low,
        "temp_max_c": float(np.max(temp)),
    }


def to_vector(features: dict[str, float]) -> np.ndarray:
    """Order a feature dict into the array the model expects.

    Ordering by FEATURE_NAMES rather than by dict insertion is the second half
    of the train/serve-skew defence: a caller who builds the dict in a different
    order still gets the columns lined up with the coefficients.
    """
    missing = [n for n in FEATURE_NAMES if n not in features]
    if missing:
        raise UnusableCharge(f"missing features: {missing}")
    return np.array([float(features[n]) for n in FEATURE_NAMES], dtype=float)
