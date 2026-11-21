import os
import pandas as pd
from src.logger import get_console_logger
from typing import Optional

logger = get_console_logger('Data-ingestion')

## The data import from API is also imported here if there is any..
def ingest_data(file_path: Optional[str] = None)-> pd.DataFrame:
    if file_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, 'data', 'data.csv')
    data = pd.read_csv(file_path)
    return data