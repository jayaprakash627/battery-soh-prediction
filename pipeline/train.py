"""Fit the model, and — the actual point of this file — measure it honestly.

Three ways to score the same model
----------------------------------
The number a battery SoH project reports depends far more on how it splits the
data than on which algorithm it uses. This script runs all three splits and
prints them side by side, because the gap between them *is* the finding.

  random        Shuffle every cycle from every cell and hold out 20%.
                This is the split most people report, and it is close to
                meaningless here. Cycle 100 and cycle 101 of the same cell are
                nearly the same measurement; with one in train and one in test,
                the model can score well by interpolating between neighbours it
                has already seen. It answers "can the model fill in a gap in an
                experiment it has already watched" — a question nobody has.

  by_cell       Leave-one-cell-out. Train on every cell but one, predict that
                one, rotate. The held-out cell's manufacturing spread and its
                whole degradation trajectory are unseen. This is the number
                that corresponds to putting the model on a pack it has never
                met, so this is the number that decides which model ships.

  forward_time  Train on the first 60% of every cell's life, test on the last
                40%. Held-out *degradation levels*, not held-out cells: it asks
                whether the model extrapolates into damage it has never seen, or
                only interpolates within it. A fleet tool is most needed on old
                packs, which is exactly where this split is hardest.

With eight cells, `by_cell` has eight folds and the smallest holds 39 cycles.
The spread across folds is reported alongside the mean, because a mean of eight
numbers that range over two points is a different claim than one that does not.

Run:  python -m pipeline.train
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from app.features import COLLINEARITY_THRESHOLD, CORRELATION_WARNING, FEATURE_NAMES
from pipeline.build_dataset import load_table

RANDOM_SEED = 20260815  # fixed so a rerun reproduces the numbers in the README

# Candidate models. Kept deliberately small and shallow: with ~900 rows from
# eight cells, a deep forest will memorise the eight trajectories and score
# beautifully on the random split while learning nothing transferable.
CANDIDATES = {
    "ridge": lambda: Ridge(alpha=1.0),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        min_samples_leaf=5, random_state=RANDOM_SEED),
}


def _standardise(train: np.ndarray, *others: np.ndarray):
    """Centre and scale on the training fold only.

    Fitting the scaler on all the data before splitting is the quietest form of
    leakage there is: the test fold's mean and spread bleed into the training
    features, every score improves slightly, and nothing looks wrong. Computing
    it inside the fold costs three lines and removes the doubt.
    """
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0  # a constant feature contributes nothing, not NaN
    return ((train - mean) / scale,
            *[(o - mean) / scale for o in others],
            mean, scale)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        # R² is reported because everyone expects it, but MAE is the number
        # that means something to an operator: it is in percentage points of
        # state of health, the same unit as the answer on the screen.
        "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else float("nan"),
        "mae_pct": round(float(np.mean(np.abs(err))), 4),
        "rmse_pct": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "max_abs_err_pct": round(float(np.max(np.abs(err))), 4),
        "n": int(y_true.size),
    }


def _fit_predict(make_model, X_tr, y_tr, X_te):
    X_tr_s, X_te_s, _, _ = _standardise(X_tr, X_te)
    model = make_model()
    model.fit(X_tr_s, y_tr)
    return model.predict(X_te_s)


def evaluate_random(make_model, X, y, folds=5) -> dict:
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        preds[te] = _fit_predict(make_model, X[tr], y[tr], X[te])
    return _metrics(y, preds)


def evaluate_by_cell(make_model, X, y, groups) -> tuple[dict, dict, np.ndarray]:
    """Leave-one-cell-out. Returns (pooled metrics, per-cell metrics, predictions)."""
    preds = np.zeros_like(y)
    per_cell: dict[str, dict] = {}
    for cell in sorted(set(groups)):
        te = groups == cell
        tr = ~te
        preds[te] = _fit_predict(make_model, X[tr], y[tr], X[te])
        per_cell[cell] = _metrics(y[te], preds[te])
    pooled = _metrics(y, preds)
    pooled["mae_spread_across_cells_pct"] = round(
        max(m["mae_pct"] for m in per_cell.values())
        - min(m["mae_pct"] for m in per_cell.values()), 4)
    return pooled, per_cell, preds


def evaluate_forward_time(make_model, X, y, groups, rows, train_frac=0.6) -> dict:
    """Early life -> late life, split within each cell by cycle number."""
    cycles = np.array([int(r["cycle_number"]) for r in rows])
    train_mask = np.zeros(len(y), dtype=bool)
    for cell in sorted(set(groups)):
        sel = np.flatnonzero(groups == cell)
        cutoff = np.quantile(cycles[sel], train_frac)
        train_mask[sel[cycles[sel] <= cutoff]] = True
    te = ~train_mask
    preds = _fit_predict(make_model, X[train_mask], y[train_mask], X[te])
    out = _metrics(y[te], preds)
    out["train_n"] = int(train_mask.sum())
    return out


def collinear_pairs(X: np.ndarray) -> list[dict]:
    """Feature pairs correlated above the threshold, measured not assumed.

    Recorded in the artifact so the API can publish it. The point is that a
    reader does not have to take the caveat in features.py on trust — the
    numbers behind it ship with the model, and if a future dataset makes the
    features independent, this list empties out on its own.
    """
    if X.shape[0] < 3:
        return []
    corr = np.corrcoef(X.T)
    out = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            if abs(corr[i, j]) >= COLLINEARITY_THRESHOLD:
                out.append({"a": FEATURE_NAMES[i], "b": FEATURE_NAMES[j],
                            "r": round(float(corr[i, j]), 3)})
    return sorted(out, key=lambda d: -abs(d["r"]))


def export_ridge(model: Ridge, mean: np.ndarray, scale: np.ndarray,
                 residuals: np.ndarray) -> dict:
    """Serialise a fitted ridge model to plain JSON.

    Not a pickle. Three reasons, in order of how much they would hurt:
    unpickling executes code, so a model file becomes an attack surface the
    moment anyone can replace it; a pickle is bound to the exact scikit-learn
    that wrote it and breaks on upgrade with an unreadable traceback; and JSON
    means the *serving* code needs numpy and nothing else — no scikit-learn, no
    scipy, on the deployed box. The whole model is eight coefficients. It should
    be readable by a human, and it is.
    """
    # Empirical prediction interval from the held-out-cell residuals, not from
    # any Gaussian assumption. The claim it supports is exactly what was
    # measured: on cells the model had never seen, this fraction of predictions
    # landed within this band.
    abs_res = np.abs(residuals)
    return {
        "kind": "ridge",
        "feature_names": list(FEATURE_NAMES),
        "coefficients": [float(c) for c in model.coef_],
        "intercept": float(model.intercept_),
        "standardiser": {
            "mean": [float(m) for m in mean],
            "scale": [float(s) for s in scale],
        },
        "interval": {
            "p50_abs_err_pct": round(float(np.percentile(abs_res, 50)), 4),
            "p90_abs_err_pct": round(float(np.percentile(abs_res, 90)), 4),
            "p95_abs_err_pct": round(float(np.percentile(abs_res, 95)), 4),
            "basis": "leave-one-cell-out residuals",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    X, y, groups, rows = load_table(args.processed / "features.csv")
    print(f"{len(y)} cycles, {len(set(groups))} cells, "
          f"SoH {y.min():.1f}%–{y.max():.1f}%\n")

    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": RANDOM_SEED,
        "n_cycles": int(len(y)),
        "cells": sorted(set(groups)),
        "features": list(FEATURE_NAMES),
        "models": {},
    }

    loco_preds: dict[str, np.ndarray] = {}
    for name, make in CANDIDATES.items():
        random_m = evaluate_random(make, X, y)
        by_cell_m, per_cell, preds = evaluate_by_cell(make, X, y, groups)
        forward_m = evaluate_forward_time(make, X, y, groups, rows)
        loco_preds[name] = preds

        report["models"][name] = {
            "random": random_m,
            "by_cell": by_cell_m,
            "by_cell_per_fold": per_cell,
            "forward_time": forward_m,
            # The headline of the whole project: how much the optimistic split
            # flatters the model, in percentage points of MAE.
            "optimism_gap_mae_pct": round(
                by_cell_m["mae_pct"] - random_m["mae_pct"], 4),
        }
        print(f"{name:>18}  random MAE {random_m['mae_pct']:5.2f}  |  "
              f"by-cell MAE {by_cell_m['mae_pct']:5.2f} (R² {by_cell_m['r2']:.3f})  |  "
              f"forward-time MAE {forward_m['mae_pct']:5.2f}")

    # Selection is on by-cell MAE and nothing else. Chosen before the numbers
    # were seen, and written down here so the choice cannot drift to whichever
    # split happens to favour a preferred model.
    winner = min(CANDIDATES, key=lambda n: report["models"][n]["by_cell"]["mae_pct"])
    report["selected_model"] = winner
    report["selection_rule"] = "lowest leave-one-cell-out MAE"
    print(f"\nselected: {winner}")

    # Ship ridge if it is within a quarter of a percentage point of the best.
    # A linear model over eight physically-meaningful features can state which
    # measurement moved the answer and by how much; a forest can only gesture at
    # importances. For a tool whose output is "replace this pack", being able to
    # say *why* is worth more than 0.25 points of MAE — and a tie-break rule
    # written down in advance is a decision, whereas one applied afterwards is
    # an excuse.
    ridge_mae = report["models"]["ridge"]["by_cell"]["mae_pct"]
    best_mae = report["models"][winner]["by_cell"]["mae_pct"]
    shipped = "ridge" if (ridge_mae - best_mae) <= 0.25 else winner
    report["shipped_model"] = shipped
    report["ship_rule"] = (
        "ridge unless another model beats it by more than 0.25 points of "
        "by-cell MAE; ridge is explainable per-feature and serves without "
        "scikit-learn"
    )
    if shipped != winner:
        print(f"shipping: {shipped} (ridge was {ridge_mae - best_mae:+.2f} MAE)")

    if shipped != "ridge":
        raise SystemExit(
            f"{shipped} beat ridge by more than the 0.25-point threshold. "
            f"Serving a tree ensemble needs a tree exporter that this repo "
            f"does not have yet — see README, 'What I would do next'."
        )

    # Final fit on every cell, standardiser included, for the served model.
    X_s, mean, scale = _standardise(X)
    final = Ridge(alpha=1.0)
    final.fit(X_s, y)

    residuals = loco_preds["ridge"] - y
    artifact = export_ridge(final, mean, scale, residuals)
    artifact["trained_on"] = {
        "cycles": int(len(y)), "cells": sorted(set(groups)),
        "soh_range_pct": [round(float(y.min()), 2), round(float(y.max()), 2)],
    }
    artifact["honest_performance"] = report["models"]["ridge"]["by_cell"]
    # The range each measurement actually took across the training data. The UI
    # uses it to bound its "what if" sliders, so a slider cannot be dragged to a
    # value no real battery has ever produced. p1/p99 rather than min/max, so one
    # freak charge does not stretch a slider into mostly-empty space.
    artifact["feature_ranges"] = {
        name: {
            "low": round(float(np.percentile(X[:, i], 1)), 6),
            "high": round(float(np.percentile(X[:, i], 99)), 6),
            "typical": round(float(np.median(X[:, i])), 6),
        }
        for i, name in enumerate(FEATURE_NAMES)
    }
    artifact["collinearity"] = {
        "threshold": COLLINEARITY_THRESHOLD,
        "pairs": collinear_pairs(X),
        "caveat": CORRELATION_WARNING,
    }

    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "model.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (args.artifacts / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")

    # Per-cycle predictions for the UI's "predicted vs measured over life" chart.
    # These are the leave-one-cell-out predictions, so every point on that chart
    # comes from a model that had not seen that cell.
    curve = [
        {"cell_id": r["cell_id"], "cycle": int(r["cycle_number"]),
         "measured_soh_pct": round(float(y[i]), 3),
         "predicted_soh_pct": round(float(loco_preds["ridge"][i]), 3)}
        for i, r in enumerate(rows)
    ]
    (args.artifacts / "loco_predictions.json").write_text(
        json.dumps(curve, indent=1) + "\n")

    print(f"\n-> {args.artifacts / 'model.json'}")
    print(f"-> {args.artifacts / 'evaluation.json'}")


if __name__ == "__main__":
    main()
