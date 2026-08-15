"""Checks on the committed artifacts themselves.

The model that ships is a file in this repository, so it can go stale in ways no
amount of testing the *code* would catch — someone edits app/features.py and
forgets to retrain, and the service happily multiplies the new features by the
old coefficients. These tests fail when the committed artifacts and the current
code have drifted apart.

They skip rather than fail on a clone that has not run the pipeline yet, since
the raw data is a 209 MB download and requiring it to run the suite would mean
nobody runs the suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.features import FEATURE_NAMES, extract
from app.model import Model

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "model.json").exists(),
    reason="artifacts not built; run the pipeline (see README)",
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads((ARTIFACTS / "model.json").read_text())


def test_committed_model_matches_the_current_feature_code(artifact):
    """The drift tripwire. If this fails, retrain — do not edit the JSON."""
    assert tuple(artifact["feature_names"]) == FEATURE_NAMES
    Model(artifact)  # constructing it re-checks the same thing at load time


def test_committed_model_reports_its_honest_error(artifact):
    perf = artifact["honest_performance"]
    assert perf["mae_pct"] > 0
    # A leave-one-cell-out MAE that ever comes back near zero means the split
    # stopped holding cells out, not that the model became perfect.
    assert perf["mae_pct"] > 0.2
    assert artifact["interval"]["basis"] == "leave-one-cell-out residuals"


def test_committed_model_declares_its_collinearity(artifact):
    """The caveat has to ship with the numbers behind it, not just as prose."""
    collinearity = artifact["collinearity"]
    assert collinearity["caveat"]
    assert collinearity["pairs"], "expected the known correlated feature group"
    assert all(abs(p["r"]) >= collinearity["threshold"] for p in collinearity["pairs"])


def test_evaluation_shows_the_optimism_gap(artifact):
    report = json.loads((ARTIFACTS / "evaluation.json").read_text())
    for name, m in report["models"].items():
        # Every model must do better on the shuffled split than on held-out
        # cells. If one ever did not, the by-cell split would be the thing to
        # go and check, not a result to celebrate.
        assert m["by_cell"]["mae_pct"] > m["random"]["mae_pct"], name
    assert report["shipped_model"] == "ridge"


def test_demo_curves_predict_close_to_their_measured_truth(artifact):
    """End to end on real data: the shipped model against ground truth it never saw.

    Loose bound on purpose — this is a regression guard against a broken
    pipeline, not a restatement of the accuracy claim. The real number lives in
    evaluation.json and is measured properly.
    """
    model = Model(artifact)
    demo = json.loads((ARTIFACTS / "demo_charges.json").read_text())
    errors = []
    for rows in demo["cells"].values():
        for row in rows:
            pred = model.predict(extract(**row["curve"]))
            errors.append(abs(pred.soh_pct - row["measured_soh_pct"]))
    assert errors
    # Measured at 4.3 worst / 1.5 mean when this bound was set. Left with real
    # headroom so a retrain on more cells does not fail the suite for moving a
    # point, but tight enough to catch the failure that made this test worth
    # writing: three part-charged curves that read 27 points low until the
    # run-up guard in app/features.py was fixed.
    assert max(errors) < 8, f"worst demo error {max(errors):.1f} points"
    assert sum(errors) / len(errors) < 3


def test_loco_predictions_cover_every_trained_cell(artifact):
    rows = json.loads((ARTIFACTS / "loco_predictions.json").read_text())
    assert {r["cell_id"] for r in rows} == set(artifact["trained_on"]["cells"])
    assert len(rows) == artifact["trained_on"]["cycles"]


def test_admissions_record_explains_every_excluded_cell():
    """The inclusion rule has to be auditable, not just applied."""
    path = ROOT / "data" / "processed" / "admissions.json"
    if not path.exists():
        pytest.skip("ingest summary not present")
    summary = json.loads(path.read_text())
    for cell in summary["cells_excluded"]:
        assert summary["per_cell"][cell]["cell_verdict"].startswith("excluded:")
    assert summary["cells_included"]
    # The duplicated B0025–B0028 files must not have entered twice; a repeated
    # cell would sit on both sides of a leave-one-cell-out split.
    assert len(summary["cells_included"]) == len(set(summary["cells_included"]))
