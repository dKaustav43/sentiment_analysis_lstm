import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
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

def encoding_data(x_train,x_val,x_test,y_train,y_val,y_test):

    from vocab_and_encoding_functions import encode_train_text_val_dataset
    
    x_train_encoded, x_val_encoded, x_test_encoded = encode_train_text_val_dataset(
                                                                                    x_train_data = x_train,
                                                                                    x_val_data = x_val,
                                                                                    x_test_data = x_test
                                                                                    )
    y_train_encoded = np.array([1 if y == "positive" else 0 for y in y_train])
    y_val_encoded = np.array([1 if y=="positive" else 0 for y in y_val])
    y_test_encoded = np.array([1 if y=="positive" else 0 for y in y_test])

    return x_train_encoded, x_val_encoded, x_test_encoded, y_train_encoded, y_val_encoded, y_test_encoded

def converting_data_to_tensor(x_train_encoded, x_val_encoded, x_test_encoded, y_train_encoded, y_val_encoded, y_test_encoded, device):
    
    X_train_tensor = torch.tensor(x_train_encoded, dtype=torch.long, device=device)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.float, device=device)

    X_val_tensor = torch.tensor(x_val_encoded, dtype=torch.long, device=device)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.float, device=device)

    X_test_tensor = torch.tensor(x_test_encoded, dtype=torch.long, device=device)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.float, device=device)

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

def dataloader_batches(x_train_tensor,x_val_tensor,x_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor,
                        batch_size:int = 32):

    train_dataset = TensorDataset(x_train_tensor,y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor,y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor,y_test_tensor)

    train_loader = DataLoader(train_dataset,batch_size,shuffle=True)

    val_loader = DataLoader(val_dataset,batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size, shuffle = True)

    return train_loader, val_loader, test_loader

def main():
    start_time = time.perf_counter()
    df_imdb = load_imdb_data_into_df(path = path_to_data)
    

    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_set(df=df_imdb)

    end_time = time.perf_counter()
    print(f"time to load the data and split into test train and val: {end_time - start_time}s")

if __name__ == "__main__":
    main()