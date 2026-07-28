import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

path_to_data = 'data/IMDB Dataset.csv'
def load_imdb_data_into_df(path:str|Path = path_to_data) -> pd.DataFrame:
    """
        Loads the imdb data into a dataframe.
    """
    df = pd.read_csv(path)
    return df

def train_test_val_set(df:pd.DataFrame):

    """
    custom function to split the df into train,val and test set.
    """
    
    X = df['review']
    y = df['sentiment'].values

    #split for training and testing dataset (80/20)
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

    #split the training data further into training and validation dataset
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42)
    
    return x_train, x_val, x_test, y_train, y_val, y_test

def main():
    start_time = time.perf_counter()
    df_imdb = load_imdb_data_into_df(path = path_to_data)
    

    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_set(df=df_imdb)

    end_time = time.perf_counter()
    print(f"time to load the data and split into test train and val: {end_time - start_time}s")

if __name__ == "__main__":
    main()