import json
import logging
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import CONFIG

DROP_COLS = ['engine_id', 'cycle', 'max_cycle', 'RUL', 'failure']


def split_by_engine(df, test_split=None):
    """
    Split the dataframe into train/test sets by engine_id (not by row).

    Splitting by row would leak information: rows from the same engine's
    time series would appear in both train and test, letting the model
    "see" that engine's degradation pattern before being tested on it.
    Splitting by engine_id guarantees no engine appears in both sets.
    """
    if test_split is None:
        test_split = CONFIG["test_split"]

    engine_ids = sorted(df['engine_id'].unique())
    split_idx = int(len(engine_ids) * (1 - test_split))

    train_ids = set(engine_ids[:split_idx])
    test_ids = set(engine_ids[split_idx:])

    assert train_ids.isdisjoint(test_ids), "Engine leakage detected between train/test split"

    train_df = df[df['engine_id'].isin(train_ids)]
    test_df = df[df['engine_id'].isin(test_ids)]
    return train_df, test_df


def get_features_labels(df):
    X = df.drop(columns=DROP_COLS)
    y = df['failure']
    return X, y


def train_model(df, save_path='predictive_maintenance_model.pkl'):
    train_df, test_df = split_by_engine(df)

    X_train, y_train = get_features_labels(train_df)
    X_test, y_test = get_features_labels(test_df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    logging.info("Model Performance:")
    print("Accuracy:", metrics["accuracy"])
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, save_path)
    logging.info(f"Model saved as {save_path}")

    # Persist the exact feature order the model was trained on, so the
    # inference API can build a matching feature vector at prediction time.
    features_path = os.path.splitext(save_path)[0] + "_features.json"
    with open(features_path, "w") as f:
        json.dump(list(X_train.columns), f)
    logging.info(f"Feature schema saved as {features_path}")

    return model, X_test, y_test, metrics
