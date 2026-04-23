import sqlite3
import pandas as pd
import logging

def load_data(df):
    conn = sqlite3.connect('engine_data.db')
    df.to_sql('engine_data', conn, if_exists='replace', index=False)
    logging.info("Data Loaded into SQL Database")
    return conn

def query_data(conn):
    query = """
    SELECT engine_id, AVG(sensor_1) as avg_sensor_1
    FROM engine_data
    GROUP BY engine_id
    LIMIT 5
    """
    result = pd.read_sql(query, conn)
    
    logging.info("Sample SQL Query Result:")
    print(result)
