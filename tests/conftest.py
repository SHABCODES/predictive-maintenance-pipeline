import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable when running `pytest` from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def raw_columns():
    return (
        ["engine_id", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )


@pytest.fixture
def tiny_raw_file(tmp_path, raw_columns):
    """A tiny whitespace-separated file in the same format as train_FD001.txt,
    with 2 synthetic engines of different lifetimes."""
    rng = np.random.default_rng(42)
    rows = []
    for engine_id, n_cycles in [(1, 10), (2, 15)]:
        for cycle in range(1, n_cycles + 1):
            op_settings = rng.normal(size=3)
            sensors = rng.normal(loc=500, scale=5, size=21)
            rows.append([engine_id, cycle, *op_settings, *sensors])

    df = pd.DataFrame(rows, columns=raw_columns)
    file_path = tmp_path / "tiny_train.txt"
    df.to_csv(file_path, sep=" ", header=False, index=False)
    return file_path


@pytest.fixture
def transformed_df():
    """A small already-transformed dataframe (post feature engineering),
    with the columns train_model / split_by_engine expect."""
    rng = np.random.default_rng(0)
    rows = []
    for engine_id, n_cycles in [(1, 20), (2, 20), (3, 20), (4, 20), (5, 20)]:
        for cycle in range(1, n_cycles + 1):
            max_cycle = n_cycles
            rul = max_cycle - cycle
            row = {
                "engine_id": engine_id,
                "cycle": cycle,
                "max_cycle": max_cycle,
                "RUL": rul,
                "failure": 1 if rul <= 5 else 0,
            }
            for i in range(1, 6):
                row[f"sensor_{i}_rolling_mean"] = rng.normal()
                row[f"sensor_{i}_diff"] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)
