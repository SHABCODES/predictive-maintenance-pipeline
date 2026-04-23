import pandas as pd
import logging
from config import CONFIG

def transform_data(df):
    
    # Create RUL
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycle, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    
    # Rolling mean
    for col in sensor_cols:
        df[f'{col}_rolling_mean'] = df.groupby('engine_id')[col] \
                                      .rolling(CONFIG["window_size"]).mean() \
                                      .reset_index(level=0, drop=True)
    
    # Trend feature
    for col in sensor_cols:
        df[f'{col}_diff'] = df.groupby('engine_id')[col].diff()
    
    df = df.dropna()
    
    # Failure label
    df['failure'] = df['RUL'].apply(
        lambda x: 1 if x <= CONFIG["failure_threshold"] else 0
    )
    
    logging.info(f"Data Transformed: {df.shape}")
    return df
