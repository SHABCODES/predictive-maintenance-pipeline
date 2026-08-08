import json

import pytest
from fastapi.testclient import TestClient

import api as api_module
from src.model import train_model


@pytest.fixture
def client_with_model(tmp_path, transformed_df, monkeypatch):
    """Train a tiny model against the synthetic fixture data, point the API
    module at it, and return a TestClient wired up to use it."""
    model_path = tmp_path / "model.pkl"
    features_path = tmp_path / "model_features.json"

    train_model(transformed_df, save_path=str(model_path))
    # train_model derives the features path from save_path itself
    real_features_path = str(model_path).replace(".pkl", "_features.json")

    monkeypatch.setattr(api_module, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(api_module, "FEATURES_PATH", real_features_path)
    monkeypatch.setattr(api_module, "_model", None)
    monkeypatch.setattr(api_module, "_feature_columns", None)

    return TestClient(api_module.app)


def _sample_readings(n):
    reading = {
        "op_setting_1": 0.1, "op_setting_2": 0.2, "op_setting_3": 100.0,
        **{f"sensor_{i}": 500.0 + i for i in range(1, 22)},
    }
    return [reading for _ in range(n)]


def test_health_endpoint(client_with_model):
    resp = client_with_model.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_rejects_too_few_readings(client_with_model):
    payload = {"engine_id": 1, "readings": _sample_readings(2)}
    resp = client_with_model.post("/predict", json=payload)
    assert resp.status_code == 400


def test_predict_returns_valid_probability(client_with_model):
    payload = {"engine_id": 1, "readings": _sample_readings(6)}
    resp = client_with_model.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert isinstance(body["predicted_failure"], bool)
    assert body["cycles_used"] == 6
