import logging
from src.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    run_pipeline("data/train_FD001.txt")
