import os
import pandas as pd

def load_dataset(path:str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)