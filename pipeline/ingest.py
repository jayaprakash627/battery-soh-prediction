"""Turn NASA's .mat files into one flat table of cycles — and throw most of it away.

The NASA PCoE battery set is MATLAB structs nested five deep, which is a fine
way to store an experiment and a terrible way to model one. Everything
downstream reads the flat table this module writes, so the MATLAB shape is
confined to this one file.

The part that matters more than the parsing
-------------------------------------------
The archive holds 34 cells, and it is tempting to use all of them — more cells
is more data. It would also be wrong, and quietly so.

These cells were not run under one protocol. They were run under *many*,
because varying the protocol was the point of the experiment. Across the
archive the discharge rate is 1 A, 2 A or 4 A; the discharge cut-off runs from
1.84 V to 2.69 V; ambient is 4 °C, 24 °C, 43 °C or 44 °C; some cells switch
protocol partway through their own life.

That breaks the label. `Capacity` is the amp-hours delivered by *that*
discharge, so it measures the cell's health only when the discharge conditions
are held fixed. Pull 4 A instead of 2 A and you measure less capacity from an
identical cell; stop at 2.7 V instead of 1.9 V and you measure less again. A
model trained across all 34 cells would spend most of its capacity learning to
predict the experimenter's settings, and its error would look like ordinary
model noise rather than like a category error.

So this module admits a cycle only if it was run under the reference protocol
below, records every rejection with its reason in `admissions.json`, and lets
the count speak for itself. Roughly two thirds of the archive does not qualify.
Reporting a model trained on all of it would produce a bigger number and a
smaller result.

Run:  python -m pipeline.ingest --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# Rated capacity of the cells in this experiment, in amp-hours. SoH is defined
# against this rather than against each cell's own first-cycle capacity: a cell
# that left the factory under-spec should read below 100% on day one, because
# that is the truth a warranty claim turns on. Using first-cycle capacity as the
# reference would define every cell as perfectly healthy at birth by fiat.
RATED_CAPACITY_AH = 2.0

# ---------------------------------------------------------------------------
# The reference protocol.
#
# These are the conditions under which a measured capacity is a statement about
# the cell rather than about the test. The tolerances are wide enough to absorb
# instrument noise and narrow enough to exclude a different experiment: the
# archive's alternatives are far outside them (1 A and 4 A discharges against a
# 2 A reference, 4 °C and 43 °C against 24 °C), so nothing sits near a boundary
# and no cycle's fate turns on the third decimal place.
# ---------------------------------------------------------------------------
AMBIENT_C = 24.0
AMBIENT_TOLERANCE_C = 3.0

# The charge features assume a known constant current — a charger delivering
# 0.5 A traverses the voltage window at a completely different speed.
CHARGE_CURRENT_A = 1.5
CHARGE_CURRENT_TOLERANCE_A = 0.2

# The reference discharge rate. This is what makes `Capacity` comparable
# between cells at all.
DISCHARGE_CURRENT_A = 2.0
DISCHARGE_CURRENT_TOLERANCE_A = 0.25

# A 2 Ah cell that reports 2.4 Ah delivered has not exceeded physics; the
# integration has run across a protocol change or a rest period. Below 0.8 Ah
# the cell is past any definition of usable and the curve shape stops being
# comparable. Both bounds reject measurements, not cells.
CAPACITY_MIN_AH = 0.8
CAPACITY_MAX_AH = 2.2

# Per-cell gates, applied after the per-cycle ones.
MIN_CYCLES_PER_CELL = 15          # fewer is a fragment, not a trajectory
MAX_CAPACITY_TREND = -0.30        # Spearman rho of capacity against cycle order

KNOWN_TYPES = {"charge", "discharge", "impedance"}


def _vector(x) -> np.ndarray:
    """Unwrap a MATLAB row/column vector into a flat float array."""
    return np.asarray(x, dtype=float).reshape(-1)


def _fields(struct) -> set[str]:
    """Field names of a scipy mat_struct (struct_as_record=False)."""
    return set(getattr(struct, "_fieldnames", ()) or ())


def _cc_current(current: np.ndarray) -> float:
    """Representative constant-current level of a charge.

    Taken from the first third of the log, which is inside the constant-current
    phase for every charge in this archive. The median rejects the switch-on
    transient without needing to find where it ends.
    """
    head = current[: max(1, current.size // 3)]
    return float(np.median(head))


def _discharge_current(current: np.ndarray) -> float:
    """Representative discharge rate, from the middle of the discharge.

    The middle avoids both the initial transient and the tail where the load is
    released, either of which would drag a whole-log median toward zero and let
    a 4 A discharge pass as a 2 A one.
    """
    mid = current[current.size // 4: max(current.size // 4 + 1, current.size // 2)]
    return float(np.median(np.abs(mid)))


def read_cell(mat_path: Path) -> list[dict]:
    """Read one cell's .mat file into a list of operation dicts, in run order."""
    cell_id = mat_path.stem.upper()
    mat = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)

    # The top-level variable is named after the cell (`B0005`), but a stray
    # rename in a mirror of the dataset shouldn't break ingestion — so find the
    # one non-metadata variable rather than trusting the filename.
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{mat_path.name}: expected 1 variable, found {keys}")

    cycles = np.asarray(mat[keys[0]][0, 0].cycle).reshape(-1)

    ops: list[dict] = []
    for index, cyc in enumerate(cycles):
        op_type = str(np.asarray(cyc.type).squeeze().item()).strip().lower()
        if op_type not in KNOWN_TYPES:
            print(f"  ! {cell_id} op {index}: unknown type {op_type!r}, skipped",
                  file=sys.stderr)
            continue

        # Some cells log a *range* of ambient temperatures for one operation
        # (the chamber was ramping). Keep both ends; the admissibility check
        # requires the whole range to sit inside the reference band, so a cycle
        # measured while the chamber moved is rejected rather than averaged.
        ambient = _vector(cyc.ambient_temperature)
        record = {
            "cell_id": cell_id,
            "op_index": index,
            "type": op_type,
            "ambient_c": float(np.mean(ambient)) if ambient.size else float("nan"),
            "ambient_min_c": float(np.min(ambient)) if ambient.size else float("nan"),
            "ambient_max_c": float(np.max(ambient)) if ambient.size else float("nan"),
        }

        data = cyc.data[0, 0]
        present = _fields(data)

        if op_type == "impedance":
            ops.append(record)          # counted, never used — see README
            continue

        required = {"Voltage_measured", "Current_measured",
                    "Temperature_measured", "Time"}
        if not required <= present:
            print(f"  ! {cell_id} op {index} ({op_type}): missing "
                  f"{sorted(required - present)}, skipped", file=sys.stderr)
            continue

        record["voltage_v"] = _vector(data.Voltage_measured)
        record["current_a"] = _vector(data.Current_measured)
        record["temperature_c"] = _vector(data.Temperature_measured)
        record["time_s"] = _vector(data.Time)

        lengths = {len(record[k]) for k in
                   ("voltage_v", "current_a", "temperature_c", "time_s")}
        if len(lengths) != 1 or 0 in lengths:
            print(f"  ! {cell_id} op {index} ({op_type}): channel lengths "
                  f"{sorted(lengths)}, skipped", file=sys.stderr)
            continue

        if op_type == "discharge":
            # Present-but-empty happens in several of the 4 °C cells.
            capacity = _vector(data.Capacity) if "Capacity" in present else np.array([])
            if capacity.size == 0 or not np.isfinite(capacity[0]):
                continue
            record["capacity_ah"] = float(capacity[0])
            record["discharge_current_a"] = _discharge_current(record["current_a"])
            record["discharge_end_v"] = float(np.min(record["voltage_v"]))
        else:
            record["charge_current_a"] = _cc_current(record["current_a"])

        ops.append(record)

    return ops


def pair_charge_to_discharge(ops: list[dict]) -> list[dict]:
    """Attach each charge to the discharge that immediately follows it.

    This is the join the whole project rests on, so it is deliberately strict:
    a charge is paired only with the *next* discharge, and only if no other
    charge happens in between. A charge whose discharge never came (the log
    ends, or the operator ran two charges back to back) is dropped rather than
    paired with something further down the list.

    Why strict: the point of the model is to read a cell's health from a charge
    curve. Pairing a charge with a discharge that happened three operations
    later would attach a label from a different cell-state, and the error would
    look like ordinary model noise instead of like a bug. Impedance sweeps
    between the two are tolerated — they are measurements, not cycling, and the
    dataset interleaves them routinely.
    """
    paired: list[dict] = []
    pending: dict | None = None

    for op in ops:
        if op["type"] == "charge":
            # A second charge with no discharge between: the first can never be
            # labelled, so drop it and keep the newer one.
            pending = op
        elif op["type"] == "discharge":
            if pending is not None:
                paired.append({"charge": pending, "discharge": op})
                pending = None
        # impedance: leave `pending` alone.

    return paired


def admissibility(pair: dict) -> str | None:
    """Return None if the cycle was run under the reference protocol, else why not.

    A string reason rather than a bool, because the rejection counts are a
    result in their own right — they are what turns "I used eight cells" from an
    arbitrary choice into a consequence of a stated rule.
    """
    charge, discharge = pair["charge"], pair["discharge"]

    lo = min(charge["ambient_min_c"], discharge["ambient_min_c"])
    hi = max(charge["ambient_max_c"], discharge["ambient_max_c"])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "ambient temperature not recorded"
    if abs(lo - AMBIENT_C) > AMBIENT_TOLERANCE_C or abs(hi - AMBIENT_C) > AMBIENT_TOLERANCE_C:
        return f"ambient {lo:.0f}–{hi:.0f}°C outside {AMBIENT_C:.0f}±{AMBIENT_TOLERANCE_C:.0f}°C"

    if abs(charge["charge_current_a"] - CHARGE_CURRENT_A) > CHARGE_CURRENT_TOLERANCE_A:
        return (f"charge current {charge['charge_current_a']:.2f} A "
                f"≠ {CHARGE_CURRENT_A} A reference")

    if abs(discharge["discharge_current_a"] - DISCHARGE_CURRENT_A) > DISCHARGE_CURRENT_TOLERANCE_A:
        return (f"discharge rate {discharge['discharge_current_a']:.2f} A "
                f"≠ {DISCHARGE_CURRENT_A} A reference")

    capacity = discharge["capacity_ah"]
    if not (CAPACITY_MIN_AH <= capacity <= CAPACITY_MAX_AH):
        return (f"capacity {capacity:.2f} Ah outside the plausible "
                f"{CAPACITY_MIN_AH}–{CAPACITY_MAX_AH} Ah band")

    return None


def _spearman(values: np.ndarray) -> float:
    """Rank correlation of a series against its own order.

    Written out rather than imported so this check is readable at the point it
    is used — it is three lines and the whole cell-level gate turns on it.
    """
    n = values.size
    if n < 4:
        return float("nan")
    order = np.argsort(np.argsort(values)).astype(float)
    position = np.arange(n, dtype=float)
    order -= order.mean()
    position -= position.mean()
    denom = np.sqrt((order ** 2).sum() * (position ** 2).sum())
    return float((order * position).sum() / denom) if denom else float("nan")


def build(raw_dir: Path, out_dir: Path) -> dict:
    """Read every .mat under raw_dir, pair and filter, save an .npz."""
    # The archive ships B0025–B0028 twice, in two different sub-archives, with
    # identical contents. Keyed by cell id so the duplicate cannot enter the
    # table twice and make one cell look like two independent ones — which
    # would put the *same* cell on both sides of a leave-one-cell-out split and
    # silently undo the entire point of that evaluation.
    by_id: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(raw_dir.rglob("*.mat")):
        if not path.stem.upper().startswith("B"):
            continue
        cell_id = path.stem.upper()
        if cell_id in by_id:
            duplicates.append(str(path.relative_to(raw_dir)))
            continue
        by_id[cell_id] = path

    if not by_id:
        raise SystemExit(
            f"No B*.mat files under {raw_dir}. See the README for how to fetch "
            f"the NASA dataset."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict] = []
    admissions: dict[str, dict] = {}
    global_reasons: Counter[str] = Counter()

    for cell_id, path in sorted(by_id.items()):
        ops = read_cell(path)
        pairs = pair_charge_to_discharge(ops)

        kept, reasons = [], Counter()
        for pair in pairs:
            reason = admissibility(pair)
            if reason is None:
                kept.append(pair)
            else:
                # Bucket by the leading words so the same rule with different
                # numbers in it groups into one line.
                reasons[" ".join(reason.split()[:3])] += 1

        record = {
            "file": str(path.relative_to(raw_dir)),
            "paired_cycles": len(pairs),
            "admissible_cycles": len(kept),
            "rejected": dict(reasons),
        }
        global_reasons.update(reasons)

        if len(kept) < MIN_CYCLES_PER_CELL:
            record["cell_verdict"] = (
                f"excluded: {len(kept)} admissible cycles, "
                f"fewer than the {MIN_CYCLES_PER_CELL} needed to be a trajectory")
            admissions[cell_id] = record
            print(f"  {cell_id}: excluded ({len(kept)}/{len(pairs)} admissible)")
            continue

        capacities = np.array([p["discharge"]["capacity_ah"] for p in kept])
        trend = _spearman(capacities)
        record["capacity_trend_rho"] = round(trend, 3)
        record["capacity_first_ah"] = round(float(capacities[0]), 4)
        record["capacity_last_ah"] = round(float(capacities[-1]), 4)

        if not (trend <= MAX_CAPACITY_TREND):
            # A cell whose measured capacity does not fall over its life is not
            # producing a valid capacity measurement — something changed that
            # the protocol filter did not catch. Rejected on the *label*, before
            # any model is fitted, so this cannot be tuning for a nicer score.
            record["cell_verdict"] = (
                f"excluded: capacity trend rho={trend:+.2f}, not the decline a "
                f"cycling cell must show (need <= {MAX_CAPACITY_TREND})")
            admissions[cell_id] = record
            print(f"  {cell_id}: excluded (capacity trend rho={trend:+.2f})")
            continue

        record["cell_verdict"] = "included"
        record["soh_first_pct"] = round(100 * float(capacities[0]) / RATED_CAPACITY_AH, 2)
        record["soh_last_pct"] = round(100 * float(capacities[-1]) / RATED_CAPACITY_AH, 2)
        admissions[cell_id] = record

        for n, pair in enumerate(kept, start=1):
            # The cell's equivalent-cycle count. Recorded for plotting and for
            # the forward-in-time split; never given to the model — see
            # app/features.py for why that would be cheating.
            pair["cycle_number"] = n
            pair["cell_id"] = cell_id
        all_pairs.extend(kept)

        print(f"  {cell_id}: {len(kept)}/{len(pairs)} admissible, "
              f"SoH {record['soh_first_pct']:.0f}% → {record['soh_last_pct']:.0f}%, "
              f"rho={trend:+.2f}")

    included = [c for c, r in admissions.items() if r["cell_verdict"] == "included"]
    if not included:
        raise SystemExit("No cell passed the reference-protocol filter.")

    summary = {
        "reference_protocol": {
            "ambient_c": [AMBIENT_C - AMBIENT_TOLERANCE_C, AMBIENT_C + AMBIENT_TOLERANCE_C],
            "charge_current_a": [CHARGE_CURRENT_A - CHARGE_CURRENT_TOLERANCE_A,
                                 CHARGE_CURRENT_A + CHARGE_CURRENT_TOLERANCE_A],
            "discharge_current_a": [DISCHARGE_CURRENT_A - DISCHARGE_CURRENT_TOLERANCE_A,
                                    DISCHARGE_CURRENT_A + DISCHARGE_CURRENT_TOLERANCE_A],
            "capacity_band_ah": [CAPACITY_MIN_AH, CAPACITY_MAX_AH],
            "min_cycles_per_cell": MIN_CYCLES_PER_CELL,
            "max_capacity_trend_rho": MAX_CAPACITY_TREND,
        },
        "cells_in_archive": len(by_id),
        "duplicate_files_skipped": duplicates,
        "cells_included": sorted(included),
        "cells_excluded": sorted(set(admissions) - set(included)),
        "cycles_admitted": len(all_pairs),
        "rejection_reasons": dict(global_reasons),
        "per_cell": admissions,
    }

    # Saved as an object array: the per-cycle curves are ragged (a charge near
    # end-of-life has fewer samples than a fresh one), so a rectangular array
    # would mean padding, and padding means someone eventually averages over it.
    np.savez_compressed(out_dir / "cycles.npz",
                        pairs=np.array(all_pairs, dtype=object),
                        rated_capacity_ah=RATED_CAPACITY_AH)
    (out_dir / "admissions.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{len(all_pairs)} admissible cycles from {len(included)} of "
          f"{len(by_id)} cells -> {out_dir / 'cycles.npz'}")
    print(f"included: {', '.join(sorted(included))}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    build(args.raw, args.out)


if __name__ == "__main__":
    main()
