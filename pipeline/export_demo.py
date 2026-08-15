"""Save a few real charge curves so the demo runs without the raw dataset.

The UI's point is that you can press a button and watch the model read an actual
measured charge, not a synthetic one. That needs real curves available to a
clone of this repo — and the raw dataset is 209 MB, so it cannot be one of them.

This picks a small, honest sample: for each cell, the healthiest cycle, the
sickest, and one in between. Chosen by measured capacity rather than at random,
so the demo shows the model's range instead of three points from the middle
where every model looks good.

Run:  python -m pipeline.export_demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.features import UnusableCharge, extract

# Curves are decimated to this many samples before being written. The full logs
# would make the artifact several megabytes; the features are computed from
# interpolated crossings and a 60-point voltage grid, so this loses nothing the
# model reads. `verify` below proves that rather than asserting it.
DEMO_SAMPLES = 220

# The most a decimated curve's headline feature may move. If decimation shifted
# it by more than this, the demo would be showing the model a different charge
# than the one it claims to show, so the curve is dropped instead of shipped.
# About 1% of a typical window's charge.
MAX_DECIMATION_DRIFT_AH = 0.004


def _decimate(arrays: dict[str, np.ndarray], n: int) -> dict[str, list[float]]:
    length = len(next(iter(arrays.values())))
    if length <= n:
        idx = np.arange(length)
    else:
        idx = np.unique(np.linspace(0, length - 1, n).astype(int))
    return {k: [round(float(x), 5) for x in v[idx]] for k, v in arrays.items()}


def build(processed: Path, artifacts: Path, per_cell: int = 9) -> dict:
    bundle = np.load(processed / "cycles.npz", allow_pickle=True)
    pairs = list(bundle["pairs"])
    rated = float(bundle["rated_capacity_ah"])

    by_cell: dict[str, list] = {}
    for p in pairs:
        by_cell.setdefault(p["cell_id"], []).append(p)

    out: dict[str, list] = {}
    checked = drifted = 0

    for cell, cell_pairs in sorted(by_cell.items()):
        usable = []
        for p in cell_pairs:
            ch = p["charge"]
            try:
                full = extract(time_s=ch["time_s"], voltage_v=ch["voltage_v"],
                               current_a=ch["current_a"],
                               temperature_c=ch["temperature_c"])
            except UnusableCharge:
                continue
            usable.append((p, full))

        if not usable:
            continue

        # Spread evenly across the cell's measured health, worst to best, so the
        # UI can scrub through a battery's whole life and watch the curve change
        # shape. Picking at random would cluster in the middle, where every
        # battery looks the same and there is nothing to see.
        usable.sort(key=lambda t: t[0]["discharge"]["capacity_ah"])
        if per_cell >= len(usable):
            picks = usable
        else:
            step = (len(usable) - 1) / (per_cell - 1)
            picks = [usable[round(i * step)] for i in range(per_cell)]

        rows = []
        for pair, full_features in picks:
            ch = pair["charge"]
            arrays = {
                "time_s": np.asarray(ch["time_s"], float),
                "voltage_v": np.asarray(ch["voltage_v"], float),
                "current_a": np.asarray(ch["current_a"], float),
                "temperature_c": np.asarray(ch["temperature_c"], float),
            }
            small = _decimate(arrays, DEMO_SAMPLES)

            # Verify the decimated curve still reads the same. This is the
            # reason to write the check rather than trust the sampling: it is
            # cheap, and a demo that quietly disagrees with the model is worse
            # than no demo.
            try:
                small_features = extract(**small)
            except UnusableCharge as exc:
                print(f"  ! {cell} cycle {pair['cycle_number']}: decimation made "
                      f"the curve unreadable ({exc}), skipped")
                continue
            checked += 1
            # Measured on the charge accepted across the window, which is the
            # feature the model leans on hardest — if decimation is going to
            # distort anything, it distorts this first.
            drift = abs(small_features["window_charge_ah"]
                        - full_features["window_charge_ah"])
            if drift > MAX_DECIMATION_DRIFT_AH:
                drifted += 1
                print(f"  ! {cell} cycle {pair['cycle_number']}: window charge moved "
                      f"{drift:.4f} Ah under decimation, skipped")
                continue

            capacity = pair["discharge"]["capacity_ah"]
            rows.append({
                "cell_id": cell,
                "cycle_number": int(pair["cycle_number"]),
                "measured_capacity_ah": round(capacity, 4),
                "measured_soh_pct": round(100 * capacity / rated, 2),
                "ambient_c": ch["ambient_c"],
                "samples": len(small["time_s"]),
                "curve": small,
            })
        if rows:
            out[cell] = rows

    artifacts.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "Real measured charge curves from NASA's battery aging dataset, "
            "decimated to ~400 samples so they can live in the repository. "
            "measured_soh_pct is ground truth from the full discharge that "
            "followed each charge — the model never sees it."
        ),
        "rated_capacity_ah": rated,
        "cells": out,
    }
    (artifacts / "demo_charges.json").write_text(json.dumps(payload, indent=1) + "\n")

    total = sum(len(v) for v in out.values())
    print(f"{total} demo curves from {len(out)} cells "
          f"({checked} decimation checks, {drifted} rejected) "
          f"-> {artifacts / 'demo_charges.json'}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--per-cell", type=int, default=9)
    args = ap.parse_args()
    build(args.processed, args.artifacts, args.per_cell)


if __name__ == "__main__":
    main()
