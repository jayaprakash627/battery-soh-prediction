"""The HTTP surface, including the refusals.

The interesting tests here are the ones about *not* answering. A service that
returns 200 with a guess when it cannot read the input is worse than one that
errors, because the caller has no way to tell the two apart — and the caller in
this case is deciding whether to replace a battery pack.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app import model as model_module
from tests.synthetic import charge_curve


@pytest.fixture
def client(model_artifact, monkeypatch):
    model_module.load.cache_clear()
    monkeypatch.setattr(model_module, "ARTIFACT_PATH", model_artifact)
    yield TestClient(main_module.app)
    model_module.load.cache_clear()


def test_predict_returns_an_estimate_with_an_interval(client):
    r = client.post("/api/predict", json=charge_curve(capacity_ah=1.75))
    assert r.status_code == 200
    p = r.json()["prediction"]
    assert 40 < p["soh_pct"] < 120
    assert p["interval_low_pct"] < p["soh_pct"] < p["interval_high_pct"]
    assert len(p["contributions"]) == 7


def test_predict_echoes_the_features_it_used(client):
    """So a caller can audit the answer without re-implementing extraction."""
    r = client.post("/api/predict", json=charge_curve(capacity_ah=1.75))
    features = r.json()["features"]
    assert features["window_charge_ah"] > 0
    assert set(features) == {c["feature"] for c in r.json()["prediction"]["contributions"]}


def test_unreadable_charge_is_422_with_a_reason(client):
    """Not a 200 with a guess, and not a bare 500."""
    r = client.post("/api/predict", json=charge_curve(capacity_ah=1.8, start_soc=0.9))
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "unusable_charge"
    assert "already part" in detail["reason"]
    assert detail["help"]


def test_empty_channel_is_rejected_by_validation(client):
    curve = charge_curve()
    curve["voltage_v"] = []
    assert client.post("/api/predict", json=curve).status_code == 422


def test_oversized_payload_is_rejected(client):
    curve = charge_curve()
    curve["time_s"] = [float(i) for i in range(main_module.MAX_SAMPLES + 1)]
    assert client.post("/api/predict", json=curve).status_code == 422


def test_bad_confidence_is_a_400(client):
    r = client.post("/api/predict", json={**charge_curve(), "confidence": 0.42})
    assert r.status_code == 400


def test_model_card_publishes_the_honest_number_and_the_limits(client):
    card = client.get("/api/model").json()
    assert card["honest_performance"]["mae_pct"] > 0
    assert len(card["features"]) == 7
    assert all(f["physics"] for f in card["features"])
    # The limitations are not decoration — the card is the artefact someone
    # reads before trusting this, so an empty list should fail the build.
    assert len(card["limitations"]) >= 4
    assert card["end_of_life_threshold_pct"] == model_module.END_OF_LIFE_SOH_PCT


def test_health_reports_whether_the_model_actually_loaded(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_degrades_rather_than_crashes_without_a_model(monkeypatch, tmp_path):
    model_module.load.cache_clear()
    monkeypatch.setattr(model_module, "ARTIFACT_PATH", tmp_path / "absent.json")
    body = TestClient(main_module.app).get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    model_module.load.cache_clear()


def test_predict_is_503_not_500_when_the_model_is_missing(monkeypatch, tmp_path):
    model_module.load.cache_clear()
    monkeypatch.setattr(model_module, "ARTIFACT_PATH", tmp_path / "absent.json")
    r = TestClient(main_module.app).post("/api/predict", json=charge_curve())
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "model_not_built"
    model_module.load.cache_clear()


def test_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "state of health" in r.text.lower()


def test_a_healthier_charge_scores_higher_over_http(client):
    """End to end, the API preserves the direction the physics implies."""
    good = client.post("/api/predict", json=charge_curve(capacity_ah=2.0)).json()
    bad = client.post("/api/predict", json=charge_curve(capacity_ah=1.4)).json()
    assert bad["prediction"]["soh_pct"] < good["prediction"]["soh_pct"]


def test_predict_from_features_matches_predicting_from_the_curve(client):
    """The two entry points must never disagree — they are one model."""
    from_curve = client.post("/api/predict", json=charge_curve(capacity_ah=1.7)).json()
    from_features = client.post(
        "/api/predict/features",
        json={"features": from_curve["features"], "confidence": 0.9}).json()
    assert from_features["prediction"]["soh_pct"] == \
        from_curve["prediction"]["soh_pct"]


def test_predict_from_features_rejects_a_missing_measurement(client):
    r = client.post("/api/predict/features", json={"features": {"temp_max_c": 25.0}})
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "missing_features"


def test_predict_from_features_rejects_an_invented_measurement(client):
    from app.features import FEATURE_NAMES
    features = {n: 1.0 for n in FEATURE_NAMES}
    features["battery_vibes"] = 9.0
    r = client.post("/api/predict/features", json={"features": features})
    assert r.status_code == 422
    assert r.json()["detail"]["unknown"] == ["battery_vibes"]
