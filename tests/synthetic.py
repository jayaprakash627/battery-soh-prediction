"""A fake battery, for tests only.

Nothing here is used to train or evaluate anything — the model in this repo is
fitted on measured NASA data and only on that. This exists so the tests can ask
questions the real dataset cannot answer cleanly:

  "If capacity drops and *nothing else changes*, does the window time fall?"

In real data every property moves at once, so a test written against it can only
check correlation. Here capacity and internal resistance are dials, which makes
it possible to test the feature extraction the way you test any other function:
one input at a time, with a known right answer.
"""

from __future__ import annotations

import numpy as np


def charge_curve(
    capacity_ah: float = 2.0,
    resistance_ohm: float = 0.05,
    current_a: float = 1.5,
    v_limit: float = 4.2,
    start_soc: float = 0.05,
    ambient_c: float = 24.0,
    sample_seconds: float = 5.0,
    seed: int | None = None,
) -> dict[str, list[float]]:
    """A plausible CC-CV charge, in the shape app.features.extract expects.

    Constant current until the terminal voltage hits the limit, then constant
    voltage with an exponentially decaying current. Terminal voltage is an
    open-circuit curve plus an IR drop, so `resistance_ohm` shifts the whole
    curve upward exactly the way ageing does in the real cell.
    """
    rng = np.random.default_rng(seed)

    def ocv(soc: np.ndarray | float):
        # A smooth, monotonic stand-in for a graphite/NMC open-circuit curve:
        # steep at the bottom, flat through the middle, steep again at the top.
        s = np.clip(soc, 0.0, 1.0)
        return 3.35 + 0.75 * s + 0.16 * np.tanh(6.0 * (s - 0.82))

    dt = sample_seconds
    times, volts, amps, temps = [], [], [], []

    soc = start_soc
    t = 0.0
    temp = ambient_c

    # --- constant current -------------------------------------------------
    while soc < 1.0:
        v = float(ocv(soc)) + current_a * resistance_ohm
        if v >= v_limit:
            break
        times.append(t); volts.append(v); amps.append(current_a); temps.append(temp)
        soc += (current_a * dt / 3600.0) / capacity_ah
        # I²R heating with a slow bleed to ambient. The point is only that a
        # higher resistance produces a bigger rise, which is the physics the
        # temp_rise_c feature is reading.
        temp += (current_a ** 2 * resistance_ohm) * dt * 0.010 - (temp - ambient_c) * 0.004
        t += dt
        if t > 60_000:
            raise RuntimeError("synthetic charge did not converge")

    # --- constant voltage -------------------------------------------------
    i = current_a
    while i > 0.02:
        times.append(t); volts.append(v_limit); amps.append(i); temps.append(temp)
        i *= 0.97
        soc += (i * dt / 3600.0) / capacity_ah
        temp += (i ** 2 * resistance_ohm) * dt * 0.010 - (temp - ambient_c) * 0.004
        t += dt
        if t > 120_000:
            raise RuntimeError("synthetic CV phase did not converge")

    out = {
        "time_s": np.array(times),
        "voltage_v": np.array(volts),
        "current_a": np.array(amps),
        "temperature_c": np.array(temps),
    }
    if seed is not None:
        # Measurement noise at roughly the scale of a real logger, so the tests
        # exercise the smoothing and interpolation rather than an ideal curve.
        out["voltage_v"] = out["voltage_v"] + rng.normal(0, 0.002, out["voltage_v"].shape)
        out["temperature_c"] = out["temperature_c"] + rng.normal(0, 0.05, out["temperature_c"].shape)

    return {k: v.tolist() for k, v in out.items()}
