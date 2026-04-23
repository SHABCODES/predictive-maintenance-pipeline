import pandas as pd
import logging

def extract_data(file_path):
    columns = ['engine_id', 'cycle'] + \
              [f'op_setting_{i}' for i in range(1,4)] + \
              [f'sensor_{i}' for i in range(1,22)]
    
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    df.columns = columns
    
    logging.info(f"Data Extracted: {df.shape}")
    return df
