"""Cycles in, feature table out.

Reads the paired cycles from `pipeline.ingest`, runs each charge curve through
`app.features.extract`, and writes one CSV row per usable cycle. That CSV is the
only thing training reads, and — being about 400 kB — it is committed to the
repo. Anyone can clone this and reproduce the model without downloading 209 MB
of MATLAB files.

Cycles that cannot produce honest features are dropped and *counted*, and the
count goes in the summary. A pipeline that silently discards 30% of its input is
a pipeline whose results mean something other than what they appear to.

Run:  python -m pipeline.build_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.features import FEATURE_NAMES, UnusableCharge, extract

# Columns that describe the row but are never given to the model. They exist so
# results can be grouped by cell and plotted against age; `train.py` selects
# FEATURE_NAMES explicitly and never "everything except the label", which is how
# an identifier ends up as a feature by accident.
METADATA_COLUMNS = ("cell_id", "cycle_number", "capacity_ah", "ambient_c")
LABEL_COLUMN = "soh_pct"


def build(processed_dir: Path) -> dict:
    bundle = np.load(processed_dir / "cycles.npz", allow_pickle=True)
    pairs = list(bundle["pairs"])
    rated = float(bundle["rated_capacity_ah"])

    rows: list[dict] = []
    dropped: Counter[str] = Counter()
    dropped_by_cell: Counter[str] = Counter()

    for pair in pairs:
        charge = pair["charge"]
        discharge = pair["discharge"]
        try:
            features = extract(
                time_s=charge["time_s"],
                voltage_v=charge["voltage_v"],
                current_a=charge["current_a"],
                temperature_c=charge["temperature_c"],
            )
        except UnusableCharge as exc:
            # Bucket by the first few words so near-identical reasons with
            # different numbers in them group together in the summary.
            dropped[" ".join(str(exc).split()[:4])] += 1
            dropped_by_cell[pair["cell_id"]] += 1
            continue

        rows.append({
            "cell_id": pair["cell_id"],
            "cycle_number": pair["cycle_number"],
            "capacity_ah": round(discharge["capacity_ah"], 6),
            "ambient_c": charge["ambient_c"],
            LABEL_COLUMN: round(100.0 * discharge["capacity_ah"] / rated, 4),
            **{k: round(v, 6) for k, v in features.items()},
        })

    if not rows:
        raise SystemExit("No usable cycles. Check the ingest step.")

    out_path = processed_dir / "features.csv"
    columns = list(METADATA_COLUMNS) + [LABEL_COLUMN] + list(FEATURE_NAMES)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    per_cell = Counter(r["cell_id"] for r in rows)
    summary = {
        "cycles_in": len(pairs),
        "rows_out": len(rows),
        "dropped": sum(dropped.values()),
        "drop_reasons": dict(dropped),
        "dropped_by_cell": dict(dropped_by_cell),
        "rows_by_cell": dict(per_cell),
        "soh_range_pct": [
            round(min(r[LABEL_COLUMN] for r in rows), 2),
            round(max(r[LABEL_COLUMN] for r in rows), 2),
        ],
    }
    (processed_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\n-> {out_path}")
    return summary


def load_table(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Read features.csv into (X, y, groups, rows).

    `groups` is the cell id per row, and it is what makes the honest evaluation
    in train.py possible.
    """
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    X = np.array([[float(r[name]) for name in FEATURE_NAMES] for r in rows])
    y = np.array([float(r[LABEL_COLUMN]) for r in rows])
    groups = np.array([r["cell_id"] for r in rows])
    return X, y, groups, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    build(args.processed)


if __name__ == "__main__":
    main()
