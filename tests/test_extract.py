from src.extract import extract_data


def test_extract_shape_and_columns(tiny_raw_file, raw_columns):
    df = extract_data(str(tiny_raw_file))
    assert list(df.columns) == raw_columns
    assert df.shape[0] == 25  # 10 cycles for engine 1 + 15 for engine 2


def test_extract_engine_ids_present(tiny_raw_file):
    df = extract_data(str(tiny_raw_file))
    assert set(df["engine_id"].unique()) == {1, 2}


def test_extract_no_nulls(tiny_raw_file):
    df = extract_data(str(tiny_raw_file))
    assert df.isnull().sum().sum() == 0
