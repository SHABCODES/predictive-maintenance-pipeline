import sqlite3

from src.load import load_data, query_data


def test_load_creates_table(tmp_path, transformed_df):
    db_path = tmp_path / "test.db"
    conn = load_data(transformed_df, db_path=str(db_path))

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    assert "engine_data" in tables
    conn.close()


def test_load_row_count_matches(tmp_path, transformed_df):
    db_path = tmp_path / "test.db"
    conn = load_data(transformed_df, db_path=str(db_path))

    count = conn.execute("SELECT COUNT(*) FROM engine_data").fetchone()[0]
    assert count == len(transformed_df)
    conn.close()


def test_query_data_returns_engines_ranked_by_min_rul(tmp_path, transformed_df):
    db_path = tmp_path / "test.db"
    conn = load_data(transformed_df, db_path=str(db_path))

    result = query_data(conn)
    assert "min_remaining_useful_life" in result.columns
    # results should be sorted ascending by remaining useful life
    assert list(result["min_remaining_useful_life"]) == sorted(result["min_remaining_useful_life"])
    conn.close()
