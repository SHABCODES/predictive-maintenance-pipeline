# Predictive Maintenance Data Pipeline

## Overview
This project implements an end-to-end data engineering pipeline for predictive maintenance using the NASA CMAPSS turbofan dataset.

## Features
- Modular ETL pipeline (Extract, Transform, Load)
- SQL-based data storage (SQLite)
- Time-series feature engineering (RUL, rolling mean, trend)
- Machine learning model (Random Forest)
- Leakage-free evaluation using engine-wise split

## Tech Stack
- Python
- Pandas, NumPy
- SQLite
- Scikit-learn

## Results
- Accuracy: 96%
- F1-score (failure class): 0.87

## Pipeline Flow
Raw Data → ETL → Feature Engineering → SQL → ML Model

## How to Run
```bash
pip install -r requirements.txt
python main.py
