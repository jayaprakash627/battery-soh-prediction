"""Load the trained model and answer with it — including why.

Deliberately has no scikit-learn import. The model is eight coefficients and a
standardiser in a JSON file, so serving it is a dot product; pulling in the
whole training stack to compute one would be a hundred megabytes of dependency
for arithmetic numpy already does. It also means an upgrade to scikit-learn can
never change what the deployed service predicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.features import (
    CORRELATION_WARNING,
    FEATURES,
    FEATURE_NAMES,
    to_vector,
)

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model.json"

# Below this, the pack is generally considered end-of-life for vehicle use — it
# still holds useful energy for a stationary second life, but its range is no
# longer what the vehicle was specified around. 80% is the conventional
# automotive threshold and it is a convention, not a law of nature; it is
# defined here so it can be argued with in one place.
END_OF_LIFE_SOH_PCT = 80.0

_PHYSICS = {f.name: f for f in FEATURES}


@dataclass(frozen=True)
class Contribution:
    """One feature's share of the answer, in percentage points of SoH."""

    feature: str
    value: float
    unit: str
    effect_pct: float   # how many points this feature moved the prediction
    physics: str


@dataclass(frozen=True)
class Prediction:
    soh_pct: float
    interval_low_pct: float
    interval_high_pct: float
    interval_confidence: float
    end_of_life: bool
    contributions: list[Contribution]
    baseline_pct: float
    extrapolating: bool
    notes: list[str]
    # Returned on every prediction, not only when something looks wrong. The
    # caveat applies to every answer this model gives, so hiding it behind a
    # condition would make it look like an exception rather than the rule.
    contribution_caveat: str = CORRELATION_WARNING


class Model:
    def __init__(self, artifact: dict):
        if artifact.get("kind") != "ridge":
            raise ValueError(f"unsupported model kind {artifact.get('kind')!r}")

        names = tuple(artifact["feature_names"])
        if names != FEATURE_NAMES:
            # The saved model and the running feature code disagree. Refusing is
            # the only safe move: the coefficients would still multiply cleanly
            # against a reordered vector and produce a number that looks fine.
            raise ValueError(
                "model.json was trained on different features than app/features.py "
                f"produces.\n  model: {names}\n  code:  {FEATURE_NAMES}\n"
                "Re-run `python -m pipeline.train`."
            )

        self.coef = np.asarray(artifact["coefficients"], dtype=float)
        self.intercept = float(artifact["intercept"])
        self.mean = np.asarray(artifact["standardiser"]["mean"], dtype=float)
        self.scale = np.asarray(artifact["standardiser"]["scale"], dtype=float)
        self.interval = artifact["interval"]
        self.trained_on = artifact.get("trained_on", {})
        self.honest_performance = artifact.get("honest_performance", {})
        self.collinearity = artifact.get("collinearity", {})

    def predict(self, features: dict[str, float], confidence: float = 0.90) -> Prediction:
        x = to_vector(features)
        z = (x - self.mean) / self.scale

        # Because the model is linear on standardised features, each term is
        # exactly the number of SoH points that feature contributed relative to
        # an average cell. These are not importances or approximations — they
        # sum, with the intercept, to the prediction itself.
        terms = self.coef * z
        soh = float(self.intercept + terms.sum())

        key = {0.50: "p50_abs_err_pct", 0.90: "p90_abs_err_pct",
               0.95: "p95_abs_err_pct"}.get(confidence)
        if key is None:
            raise ValueError("confidence must be one of 0.50, 0.90, 0.95")
        band = float(self.interval[key])

        notes: list[str] = []

        # Flag inputs outside the range the model was fitted on. A linear model
        # will happily extrapolate to 130% SoH; saying so is the difference
        # between a tool and a liability.
        z_max = float(np.max(np.abs(z))) if z.size else 0.0
        extrapolating = z_max > 3.0
        if extrapolating:
            worst = FEATURE_NAMES[int(np.argmax(np.abs(z)))]
            notes.append(
                f"'{worst}' is {z_max:.1f} standard deviations from anything in "
                f"the training data. This charge does not look like the cells "
                f"the model learned from, and the estimate is an extrapolation."
            )

        trained_range = self.trained_on.get("soh_range_pct")
        if trained_range and not (trained_range[0] - 5 <= soh <= trained_range[1] + 5):
            notes.append(
                f"Estimate falls outside the {trained_range[0]:.0f}–"
                f"{trained_range[1]:.0f}% band the model was trained on."
            )

        contributions = [
            Contribution(
                feature=name,
                value=float(x[i]),
                unit=_PHYSICS[name].unit,
                effect_pct=round(float(terms[i]), 3),
                physics=_PHYSICS[name].physics,
            )
            for i, name in enumerate(FEATURE_NAMES)
        ]
        contributions.sort(key=lambda c: abs(c.effect_pct), reverse=True)

        return Prediction(
            soh_pct=round(soh, 2),
            interval_low_pct=round(soh - band, 2),
            interval_high_pct=round(soh + band, 2),
            interval_confidence=confidence,
            end_of_life=soh < END_OF_LIFE_SOH_PCT,
            contributions=contributions,
            baseline_pct=round(self.intercept, 2),
            extrapolating=extrapolating,
            notes=notes,
        )


@lru_cache(maxsize=1)
def load(path: Path | None = None) -> Model:
    p = path or ARTIFACT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"No model at {p}. Build one with:\n"
            f"  python -m pipeline.ingest && python -m pipeline.build_dataset "
            f"&& python -m pipeline.train"
        )
    return Model(json.loads(p.read_text()))
