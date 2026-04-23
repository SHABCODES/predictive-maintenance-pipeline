import logging
from .extract import extract_data
from .transform import transform_data
from .load import load_data, query_data
from .model import train_model

def run_pipeline(file_path):
    logging.info("Pipeline Started")
    
    df = extract_data(file_path)
    df = transform_data(df)
    conn = load_data(df)
    query_data(conn)
    train_model(df)
    
    logging.info("Pipeline Completed Successfully ")
