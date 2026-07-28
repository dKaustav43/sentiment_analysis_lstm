import pandas as pd
import time
from pathlib import Path

path_to_data = 'data/IMDB Dataset.csv'
def load_imdb_data_into_df(path:str|Path = path_to_data) -> pd.DataFrame:
    """
        Loads the imdb data into a dataframe.
    """
    df = pd.read_csv(path)
    return df

def main():
    start_time = time.perf_counter()
    load_imdb_data_into_df(path = path_to_data)
    end_time = time.perf_counter()
    print(f"time to load the data: {end_time - start_time}s")