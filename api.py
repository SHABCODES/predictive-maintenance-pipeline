"""
Inference API for the predictive maintenance model.

Wraps the trained Random Forest classifier as a REST endpoint so it can be
called from a monitoring dashboard or automation workflow: send the last
few cycles of sensor readings for an engine, get back a failure-risk
prediction. Run the training pipeline first (`python main.py`) to produce
predictive_maintenance_model.pkl and predictive_maintenance_model_features.json.

Run locally:
    uvicorn api:app --reload

Then POST a window of readings to /predict (see README for a full example).
"""
import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import CONFIG

MODEL_PATH = "predictive_maintenance_model.pkl"
FEATURES_PATH = "predictive_maintenance_model_features.json"

app = FastAPI(
    title="Predictive Maintenance API",
    description="Failure-risk prediction for turbofan engines from recent sensor cycles.",
    version="1.0.0",
)

_model = None
_feature_columns = None


class CycleReading(BaseModel):
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    engine_id: int = Field(..., description="Identifier for the engine/asset")
    readings: list[CycleReading] = Field(
        ...,
        description=(
            f"Most recent cycles for this engine, oldest first. "
            f"Needs at least {CONFIG['window_size']} readings so rolling "
            f"features can be computed for the latest cycle."
        ),
    )


class PredictionResponse(BaseModel):
    engine_id: int
    failure_probability: float
    predicted_failure: bool
    cycles_used: int


def _load_model():
    global _model, _feature_columns
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run `python main.py` to train it first.",
            )
        _model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            _feature_columns = json.load(f)
    return _model, _feature_columns


def _build_feature_row(readings: list[CycleReading]):
    """Recreate the same rolling-mean / diff features used in training,
    for the latest cycle in the supplied window."""
    df = pd.DataFrame([r.model_dump() for r in readings])
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    for col in sensor_cols:
        df[f"{col}_rolling_mean"] = df[col].rolling(CONFIG["window_size"]).mean()
        df[f"{col}_diff"] = df[col].diff()

    latest = df.iloc[[-1]].dropna(axis=1)
    return latest


@app.get("/health")
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return {"status": "ok", "model_loaded": model_ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.readings) < CONFIG["window_size"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least {CONFIG['window_size']} cycles to compute "
                f"rolling features, got {len(request.readings)}."
            ),
        )

    model, feature_columns = _load_model()
    latest_row = _build_feature_row(request.readings)

    missing = set(feature_columns) - set(latest_row.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Could not compute required features: {sorted(missing)}",
        )

    X = latest_row[feature_columns]
    proba = float(model.predict_proba(X)[0, 1])

    return PredictionResponse(
        engine_id=request.engine_id,
        failure_probability=round(proba, 4),
        predicted_failure=proba >= 0.5,
        cycles_used=len(request.readings),
    )
