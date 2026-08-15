"""HTTP interface: send a charge curve, get a state-of-health estimate.

The API mirrors the honesty rules the model was built under. It will refuse a
charge it cannot read (422 with the reason), it returns an interval rather than
a bare number, it labels an extrapolation as one, and /api/model publishes the
model's own measured error so a caller can decide whether to trust it without
reading this repository.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app import model as model_module
from app.features import FEATURES, UnusableCharge, extract

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
ARTIFACTS = ROOT / "artifacts"

# A charge log is a few hundred to a few thousand samples. The cap is generous
# for real data and stops an unbounded array from becoming a memory problem;
# the feature extraction is O(n) but json parsing an arbitrarily large body is
# not something to leave open on a public URL.
MAX_SAMPLES = 20_000

app = FastAPI(
    title="Battery State-of-Health Prediction",
    description=(
        "Estimates remaining capacity from a single ordinary charge — no full "
        "discharge, no cycle history. Trained on NASA's battery aging dataset "
        "and scored on cells it never saw."
    ),
    version="1.0.0",
)


class ChargeCurve(BaseModel):
    """One charge event, as a charger would log it."""

    time_s: list[float] = Field(..., description="Seconds from start of charge")
    voltage_v: list[float] = Field(..., description="Measured terminal voltage")
    current_a: list[float] = Field(..., description="Measured current, positive on charge")
    temperature_c: list[float] = Field(..., description="Measured cell temperature")
    confidence: float = Field(0.90, description="Interval width: 0.50, 0.90 or 0.95")

    @field_validator("time_s", "voltage_v", "current_a", "temperature_c")
    @classmethod
    def _bounded(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("channel is empty")
        if len(v) > MAX_SAMPLES:
            raise ValueError(f"channel longer than {MAX_SAMPLES} samples")
        return v


def _prediction_payload(pred) -> dict:
    out = asdict(pred)
    out["contributions"] = [asdict(c) if not isinstance(c, dict) else c
                            for c in out["contributions"]]
    return out


@app.post("/api/predict")
def predict(curve: ChargeCurve) -> dict:
    """Estimate state of health from a raw charge curve."""
    try:
        features = extract(
            time_s=curve.time_s,
            voltage_v=curve.voltage_v,
            current_a=curve.current_a,
            temperature_c=curve.temperature_c,
        )
    except UnusableCharge as exc:
        # 422, not 200-with-a-guess. The whole point of UnusableCharge is that
        # there are charges this model must decline to read, and a caller
        # automating pack replacement needs that distinction to survive the
        # network boundary.
        raise HTTPException(status_code=422, detail={
            "error": "unusable_charge",
            "reason": str(exc),
            "help": "The model needs a charge that starts below 3.90 V and "
                    "climbs past 4.15 V, logged at 20+ samples.",
        })

    try:
        mdl = model_module.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail={
            "error": "model_not_built", "reason": str(exc)})

    try:
        pred = mdl.predict(features, confidence=curve.confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": "bad_request",
                                                     "reason": str(exc)})

    return {"features": features, "prediction": _prediction_payload(pred)}


@app.get("/api/model")
def model_card() -> dict:
    """What this model is, what it was measured at, and what it must not be used for."""
    try:
        mdl = model_module.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail={
            "error": "model_not_built", "reason": str(exc)})

    evaluation = {}
    eval_path = ARTIFACTS / "evaluation.json"
    if eval_path.exists():
        evaluation = json.loads(eval_path.read_text())

    return {
        "kind": "ridge regression on 8 physically-derived features",
        "trained_on": mdl.trained_on,
        "honest_performance": mdl.honest_performance,
        "interval": mdl.interval,
        "collinearity": mdl.collinearity,
        "end_of_life_threshold_pct": model_module.END_OF_LIFE_SOH_PCT,
        "features": [
            {"name": f.name, "unit": f.unit, "physics": f.physics,
             "direction": f.direction,
             "coefficient": mdl.coef[i]}
            for i, f in enumerate(FEATURES)
        ],
        "evaluation": {k: evaluation.get(k) for k in
                       ("selected_model", "shipped_model", "selection_rule",
                        "ship_rule", "models")},
        "limitations": [
            "Eight 18650 cells from one experiment, all at room temperature "
            "under one charge/discharge protocol. A "
            "different chemistry, format or ambient temperature is out of "
            "scope until it has been measured, not assumed.",
            "Trained between roughly 70% and 95% SoH. It has never seen a "
            "brand-new cell or one below end-of-life.",
            "It reads a charge, not a pack. A module of cells in series is a "
            "different measurement, and its weakest cell is what matters.",
            "Not a safety system. It estimates capacity, and says nothing "
            "about internal shorts, swelling or thermal runaway risk.",
            "Three of the seven features are correlated above 0.9, so the "
            "per-feature breakdown divides credit between them in a way that "
            "is not identified. The total is sound; a single bar is not a "
            "physical claim on its own.",
        ],
    }


@app.get("/api/demo/cells")
def demo_cells() -> dict:
    """Measured vs predicted across every cycle of every included cell.

    Every prediction here comes from the fold in which that cell was held out,
    so no point on this chart was made by a model that had seen the cell it is
    predicting.
    """
    path = ARTIFACTS / "loco_predictions.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail={"error": "not_built"})
    rows = json.loads(path.read_text())
    cells: dict[str, list] = {}
    for r in rows:
        cells.setdefault(r["cell_id"], []).append(r)
    return {"cells": cells}


@app.get("/api/demo/charges")
def demo_charges() -> dict:
    """A handful of real charge curves, for trying /api/predict with real data."""
    path = ARTIFACTS / "demo_charges.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail={"error": "not_built"})
    return json.loads(path.read_text())


@app.get("/api/health")
def health() -> dict:
    """Liveness plus the one thing that can actually be misconfigured here."""
    try:
        mdl = model_module.load()
        loaded, detail = True, mdl.honest_performance
    except (FileNotFoundError, ValueError) as exc:
        loaded, detail = False, str(exc)
    return {"status": "ok" if loaded else "degraded",
            "model_loaded": loaded, "model": detail}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
