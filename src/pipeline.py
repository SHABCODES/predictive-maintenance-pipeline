import logging
from .extract import extract_data
from .transform import transform_data
from .load import load_data, query_data
from .model import train_model, get_features_labels, split_by_engine
from .evaluate import evaluate_against_baseline, plot_feature_importance


def run_pipeline(file_path):
    logging.info("Pipeline Started")

    df = extract_data(file_path)
    df = transform_data(df)
    conn = load_data(df)
    query_data(conn)

    model, X_test, y_test, metrics = train_model(df)

    train_df, _ = split_by_engine(df)
    X_train, y_train = get_features_labels(train_df)

    evaluate_against_baseline(model, X_train, y_train, X_test, y_test)
    plot_feature_importance(model, list(X_test.columns))

    logging.info("Pipeline Completed Successfully")
    return metrics
