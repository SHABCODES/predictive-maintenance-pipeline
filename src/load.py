import sqlite3
import pandas as pd
import logging


def load_data(df, db_path='engine_data.db'):
    conn = sqlite3.connect(db_path)
    df.to_sql('engine_data', conn, if_exists='replace', index=False)
    logging.info("Data Loaded into SQL Database")
    return conn


def query_data(conn):
    """
    Surface engines closest to failure in the most recent cycle on record —
    the kind of query a maintenance-planning dashboard would actually run,
    rather than an arbitrary aggregate.
    """
    query = """
    SELECT engine_id, MAX(cycle) AS last_cycle, MIN(RUL) AS min_remaining_useful_life
    FROM engine_data
    GROUP BY engine_id
    ORDER BY min_remaining_useful_life ASC
    LIMIT 5
    """
    result = pd.read_sql(query, conn)

    logging.info("Engines closest to failure (lowest remaining useful life):")
    print(result)
    return result
