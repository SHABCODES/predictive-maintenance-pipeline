import pytest

from src.model import get_features_labels, split_by_engine, train_model, DROP_COLS


def test_split_by_engine_no_overlap(transformed_df):
    train_df, test_df = split_by_engine(transformed_df, test_split=0.4)
    train_ids = set(train_df["engine_id"].unique())
    test_ids = set(test_df["engine_id"].unique())
    assert train_ids.isdisjoint(test_ids)


def test_split_by_engine_covers_all_rows(transformed_df):
    train_df, test_df = split_by_engine(transformed_df, test_split=0.4)
    assert len(train_df) + len(test_df) == len(transformed_df)


def test_split_by_engine_respects_ratio(transformed_df):
    total_engines = transformed_df["engine_id"].nunique()
    train_df, test_df = split_by_engine(transformed_df, test_split=0.4)
    test_engine_count = test_df["engine_id"].nunique()
    # with 5 engines and test_split=0.4 -> 2 engines expected in test
    assert test_engine_count == int(total_engines * 0.4)


def test_get_features_labels_drops_leakage_columns(transformed_df):
    X, y = get_features_labels(transformed_df)
    for col in DROP_COLS:
        assert col not in X.columns
    assert y.name == "failure"


def test_train_model_returns_metrics_and_saves_file(tmp_path, transformed_df):
    save_path = tmp_path / "model.pkl"
    model, X_test, y_test, metrics = train_model(transformed_df, save_path=str(save_path))

    assert save_path.exists()
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(X_test) == len(y_test)
