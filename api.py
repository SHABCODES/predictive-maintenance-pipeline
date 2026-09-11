"""
Inference API for the predictive maintenance model.

Wraps the trained Random Forest classifier as a REST endpoint so it can be
called from a monitoring dashboard or automation workflow: send the last
few cycles of sensor readings for an engine, get back a failure-risk
prediction. Run the training pipeline first (`python main.py`) to produce
predictive_maintenance_model.pkl and predictive_maintenance_model_features.json.

Observability:
  - Structured JSON logs to stdout (container-native, aggregator-friendly)
  - /metrics endpoint for Prometheus scraping
  - OpenTelemetry distributed tracing (OTLP-exportable; defaults to console)
  - request_id on every response for cross-service correlation

Run locally:
    uvicorn api:app --reload

Then POST a window of readings to /predict (see README for a full example).
"""
import json
import os
import logging
import signal
import sys
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger import jsonlogger

# --- OpenTelemetry setup ---------------------------------------------------
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

_otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
if _otel_endpoint:
    # Production path: export spans to a real OTLP collector (Jaeger, Tempo, etc.)
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _exporter = OTLPSpanExporter(endpoint=_otel_endpoint)
else:
    # Default: emit spans to stdout — no external infra required
    _exporter = ConsoleSpanExporter()

_provider = TracerProvider()
_provider.add_span_processor(BatchSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer("predictive_maintenance.api")
# ---------------------------------------------------------------------------

from config import CONFIG

# --- Structured JSON logging -----------------------------------------------
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
logHandler.setFormatter(formatter)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), handlers=[logHandler])
logger = logging.getLogger("api")
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "predictive_maintenance_model.pkl")
FEATURES_PATH = os.getenv(
    "FEATURES_PATH", "predictive_maintenance_model_features.json"
)

app = FastAPI(
    title="Predictive Maintenance API",
    description="Failure-risk prediction for turbofan engines from recent sensor cycles.",
    version="1.0.0",
)

# Wire up Prometheus /metrics and OpenTelemetry auto-instrumentation
Instrumentator().instrument(app).expose(app)
FastAPIInstrumentor.instrument_app(app, tracer_provider=_provider)

# --- Graceful shutdown ------------------------------------------------------
def _shutdown_handler(sig, frame):
    """Handle SIGTERM/SIGINT so K8s rolling updates drain cleanly."""
    logger.info("shutdown_signal_received", extra={"signal": sig})
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)
# ---------------------------------------------------------------------------

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
    request_id: str = Field(
        ..., description="Unique ID for this request — use for log correlation across services"
    )


def _load_model():
    global _model, _feature_columns
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error("model_not_found", extra={"path": MODEL_PATH})
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run `python main.py` to train it first.",
            )
        logger.info("model_loading", extra={"path": MODEL_PATH})
        _model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            _feature_columns = json.load(f)
        logger.info("model_loaded")
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
    """Liveness probe — returns 200 as long as the process is alive."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe — returns 200 only after the model is loaded into memory.
    K8s will not route traffic here until this returns 200."""
    _load_model()
    return {"status": "ready", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    request_id = str(uuid.uuid4())

    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("engine_id", request.engine_id)
        span.set_attribute("readings_count", len(request.readings))
        span.set_attribute("request_id", request_id)

        if len(request.readings) < CONFIG["window_size"]:
            span.set_attribute("error", "too_few_readings")
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
            span.set_attribute("error", "missing_features")
            raise HTTPException(
                status_code=422,
                detail=f"Could not compute required features: {sorted(missing)}",
            )

        X = latest_row[feature_columns]
        proba = float(model.predict_proba(X)[0, 1])
        predicted = proba >= 0.5

        span.set_attribute("failure_probability", round(proba, 4))
        span.set_attribute("predicted_failure", predicted)

        logger.info(
            "prediction_complete",
            extra={
                "request_id": request_id,
                "engine_id": request.engine_id,
                "failure_probability": round(proba, 4),
                "predicted_failure": predicted,
            },
        )

        return PredictionResponse(
            engine_id=request.engine_id,
            failure_probability=round(proba, 4),
            predicted_failure=predicted,
            cycles_used=len(request.readings),
            request_id=request_id,
        )
