import os

from src.evaluate import evaluate_against_baseline, plot_feature_importance
from src.model import get_features_labels, split_by_engine, train_model


def test_evaluate_against_baseline_outputs(tmp_path, transformed_df):
    train_df, _ = split_by_engine(transformed_df, test_split=0.4)
    X_train, y_train = get_features_labels(train_df)

    model, X_test, y_test, _ = train_model(
        transformed_df, save_path=str(tmp_path / "model.pkl")
    )

    out_dir = tmp_path / "results"
    summary = evaluate_against_baseline(
        model, X_train, y_train, X_test, y_test, out_dir=str(out_dir)
    )

    assert "baseline_majority_class_accuracy" in summary
    assert "model_roc_auc" in summary
    assert os.path.exists(out_dir / "metrics.json")
    assert os.path.exists(out_dir / "confusion_matrix.png")
    assert os.path.exists(out_dir / "roc_curve.png")


def test_feature_importance_ranking(tmp_path, transformed_df):
    model, X_test, y_test, _ = train_model(
        transformed_df, save_path=str(tmp_path / "model.pkl")
    )
    out_dir = tmp_path / "results"
    ranked = plot_feature_importance(model, list(X_test.columns), out_dir=str(out_dir), top_n=5)

    assert len(ranked) == 5
    importances = [imp for _, imp in ranked]
    assert importances == sorted(importances, reverse=True)
    assert os.path.exists(out_dir / "feature_importance.png")
