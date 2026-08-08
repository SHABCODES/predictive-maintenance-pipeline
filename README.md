# Predictive Maintenance Pipeline

End-to-end pipeline that predicts equipment failure risk from turbofan engine sensor data (NASA C-MAPSS FD001), from raw data to a deployable prediction API.

## What it does

Raw sensor readings → automated ETL → feature engineering → SQL storage → trained classifier → evaluated against a baseline → served as a REST API for real-time failure-risk prediction.

```
data/train_FD001.txt
      │
      ▼
  extract.py        raw sensor logs → structured dataframe
      │
      ▼
  transform.py       + Remaining Useful Life (RUL) label
                      + rolling-mean & trend features per sensor
                      + failure label (RUL ≤ threshold)
      │
      ▼
  load.py             → SQLite (queryable maintenance-planning view)
      │
      ▼
  model.py             Random Forest, split BY ENGINE (no leakage)
      │
      ▼
  evaluate.py           baseline comparison, confusion matrix, ROC-AUC,
                         feature importance → results/
      │
      ▼
  api.py                FastAPI /predict endpoint for live inference
```

## Results

| Metric | Value |
|---|---|
| Model accuracy | 96.3% |
| Majority-class baseline accuracy | 86.0% |
| Model F1 (failure class) | 0.87 |
| Baseline F1 (failure class) | 0.00 |
| ROC-AUC | 0.989 |

The failure class is a minority class (~14% of rows), so a classifier that
always predicts "no failure" already scores 86% raw accuracy without
learning anything — that's the baseline row above. The model's F1 and
ROC-AUC are the numbers that actually show it learned a useful signal, not
the accuracy figure alone. Full breakdown, confusion matrix, ROC curve, and
feature importance plot are generated on each pipeline run into `results/`.

### Data leakage guard

The train/test split is done **by engine ID**, not by row. A row-wise split
would let the model see other cycles from the same engine's degradation
curve during training and be tested on it — inflating the score without
the model having learned anything generalizable. `split_by_engine()` in
`src/model.py` guarantees no engine ID appears in both sets, and this is
enforced by an assertion plus a dedicated test.

## Project structure

```
├── main.py                  # runs the full pipeline end to end
├── api.py                   # FastAPI inference service
├── config.py                # pipeline configuration (validated on import)
├── src/
│   ├── extract.py           # raw file → dataframe
│   ├── transform.py         # feature engineering
│   ├── load.py               # SQLite storage + maintenance-planning query
│   ├── model.py               # train/test split, training, leakage guard
│   └── evaluate.py            # baseline comparison, plots, metrics.json
├── tests/                    # 20 tests covering every module, incl. the API
├── results/                   # generated: metrics.json, confusion_matrix.png,
│                               roc_curve.png, feature_importance.png
├── Dockerfile
└── .github/workflows/tests.yml   # CI: runs the test suite on every push
```

## How to run

```bash
pip install -r requirements.txt

# Run the full pipeline: ETL → train → evaluate → save model + plots
python main.py

# Run the test suite (20 tests)
pytest tests/ -v

# Serve predictions
uvicorn api:app --reload
```

### Calling the prediction API

`/predict` takes the last few sensor-reading cycles for an engine (at least
5, to compute rolling features) and returns a failure-risk probability:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": 1,
    "readings": [
      {"op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0,
       "sensor_1": 518.67, "sensor_2": 642.1, "sensor_3": 1583.8,
       "sensor_4": 1396.6, "sensor_5": 14.62, "sensor_6": 21.6,
       "sensor_7": 553.9, "sensor_8": 2388.0, "sensor_9": 9046.2,
       "sensor_10": 1.3, "sensor_11": 47.2, "sensor_12": 521.7,
       "sensor_13": 2388.0, "sensor_14": 8138.6, "sensor_15": 8.42,
       "sensor_16": 0.03, "sensor_17": 392, "sensor_18": 2388,
       "sensor_19": 100.0, "sensor_20": 39.0, "sensor_21": 23.4}
    ]
  }'
```
(Include at least 5 cycles for a real prediction — this example is
truncated for readability; a 400 is returned if fewer than 5 are sent.)

### Docker

```bash
docker build -t predictive-maintenance .
docker run -p 8000:8000 predictive-maintenance
```
The image trains the model at build time, so it serves predictions as soon
as it starts. (Not build-tested in this environment — no Docker available
here — but it follows the standard pattern for a Python service.)

## Tech stack

Python, pandas, scikit-learn, SQLite, FastAPI, pytest, Docker, GitHub Actions

## Honest limitations

- Trained on a single NASA C-MAPSS subset (FD001 — one operating condition,
  one fault mode). Real plant deployments typically span multiple operating
  regimes; this hasn't been validated against FD002–FD004.
- `failure_threshold` (cycles before end-of-life counted as "failure") is a
  fixed value in `config.py`, not tuned against a cost-of-false-negative
  analysis — in a real deployment this would be set jointly with
  maintenance planning, not as a modeling default.
- The Random Forest hasn't been hyperparameter-tuned; it's the default
  100-tree configuration. A tuned model or a gradient-boosted alternative
  would likely improve on 0.87 F1, but wasn't the focus of this project.
