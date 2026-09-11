from src.extract import extract_data
from src.transform import transform_data


def test_rul_decreases_to_zero_at_last_cycle(tiny_raw_file):
    raw = extract_data(str(tiny_raw_file))
    transformed = transform_data(raw)

    for engine_id in transformed["engine_id"].unique():
        engine_rows = transformed[transformed["engine_id"] == engine_id]
        last_cycle_row = engine_rows.loc[engine_rows["cycle"].idxmax()]
        assert last_cycle_row["RUL"] == 0


def test_failure_label_matches_threshold(tiny_raw_file):
    from config import CONFIG

    raw = extract_data(str(tiny_raw_file))
    transformed = transform_data(raw)

    expected = (transformed["RUL"] <= CONFIG["failure_threshold"]).astype(int)
    assert (transformed["failure"] == expected).all()


def test_transform_drops_rolling_window_nans(tiny_raw_file):
    raw = extract_data(str(tiny_raw_file))
    transformed = transform_data(raw)
    # rolling mean / diff columns should have no NaNs left after dropna()
    assert transformed.isnull().sum().sum() == 0


def test_transform_adds_expected_feature_columns(tiny_raw_file):
    raw = extract_data(str(tiny_raw_file))
    transformed = transform_data(raw)
    assert "sensor_1_rolling_mean" in transformed.columns
    assert "sensor_1_diff" in transformed.columns
    assert "RUL" in transformed.columns
    assert "failure" in transformed.columns
