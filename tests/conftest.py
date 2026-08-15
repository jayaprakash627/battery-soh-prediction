"""Shared fixtures.

The model tests run against a *fitted-here* artifact rather than the committed
one. Two reasons: the tests then pass on a fresh clone before the pipeline has
ever been run, and a test that asserts "contributions sum to the prediction"
should be testing the serving arithmetic, not the particular coefficients that
happened to come out of the last training run.

The committed artifact is checked separately, in test_artifacts.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.features import FEATURE_NAMES, extract
from tests.synthetic import charge_curve


@pytest.fixture(scope="session")
def synthetic_table():
    """A small feature table from synthetic cells of known capacity."""
    rows = []
    for cell, resistance in enumerate([0.04, 0.05, 0.065], start=1):
        for capacity in np.linspace(2.0, 1.4, 12):
            curve = charge_curve(capacity_ah=float(capacity),
                                 resistance_ohm=resistance, seed=cell)
            rows.append({
                "cell_id": f"S{cell}",
                "soh_pct": 100.0 * float(capacity) / 2.0,
                **extract(**curve),
            })
    X = np.array([[r[n] for n in FEATURE_NAMES] for r in rows])
    y = np.array([r["soh_pct"] for r in rows])
    groups = np.array([r["cell_id"] for r in rows])
    return X, y, groups, rows


@pytest.fixture(scope="session")
def model_artifact(synthetic_table, tmp_path_factory) -> Path:
    """A ridge model fitted on the synthetic table, exported the real way."""
    from sklearn.linear_model import Ridge

    from pipeline.train import export_ridge

    X, y, _, _ = synthetic_table
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    model = Ridge(alpha=1.0).fit((X - mean) / scale, y)
    residuals = model.predict((X - mean) / scale) - y

    artifact = export_ridge(model, mean, scale, residuals)
    artifact["trained_on"] = {"cycles": len(y), "cells": ["S1", "S2", "S3"],
                              "soh_range_pct": [70.0, 100.0]}
    artifact["honest_performance"] = {"mae_pct": 1.0, "r2": 0.9}

    path = tmp_path_factory.mktemp("artifacts") / "model.json"
    path.write_text(json.dumps(artifact))
    return path


@pytest.fixture
def model(model_artifact):
    from app.model import Model
    return Model(json.loads(model_artifact.read_text()))
