import pandas as pd
import logging

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise
