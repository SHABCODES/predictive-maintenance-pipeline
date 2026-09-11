"""
Integration test for the full ETL → train → evaluate pipeline.

Exercises run_pipeline() end-to-end against a small synthetic data file,
validating that the pipeline completes and returns sensible metrics.
This closes the 0% coverage gap on src/pipeline.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_pipeline


@pytest.fixture
def pipeline_raw_file(tmp_path):
    """A tiny whitespace-separated file with two engines that have both
    failure (RUL ≤ 30) and non-failure rows, so the classifier sees both
    classes and evaluate_against_baseline can compute predict_proba correctly."""
    rng = np.random.default_rng(99)
    columns = (
        ["engine_id", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    rows = []
    # 10 engines × 40 cycles each → plenty of rows for both classes
    for engine_id in range(1, 11):
        for cycle in range(1, 41):
            op = rng.normal(size=3)
            sensors = rng.normal(loc=500, scale=5, size=21)
            rows.append([engine_id, cycle, *op, *sensors])

    df = pd.DataFrame(rows, columns=columns)
    file_path = tmp_path / "pipeline_train.txt"
    df.to_csv(file_path, sep=" ", header=False, index=False)
    return file_path


def test_pipeline_returns_metrics(pipeline_raw_file):
    """run_pipeline() should complete without error and return a metrics dict."""
    metrics = run_pipeline(str(pipeline_raw_file))
    assert isinstance(metrics, dict), "run_pipeline must return a dict"
    assert "accuracy" in metrics, "metrics must contain 'accuracy'"
    assert "report" in metrics, "metrics must contain classification report"
    assert 0.0 <= metrics["accuracy"] <= 1.0, "accuracy must be in [0,1]"


def test_pipeline_produces_model_artifact(pipeline_raw_file, tmp_path, monkeypatch):
    """Pipeline should write predictive_maintenance_model.pkl to the cwd."""
    monkeypatch.chdir(tmp_path)
    run_pipeline(str(pipeline_raw_file))
    assert (tmp_path / "predictive_maintenance_model.pkl").exists(), \
        "Model .pkl was not written to disk"
    assert (tmp_path / "predictive_maintenance_model_features.json").exists(), \
        "Feature schema .json was not written to disk"
