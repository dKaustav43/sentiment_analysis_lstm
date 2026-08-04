from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, TensorDataset

from vocab_and_encoding import encode_train_text_val_dataset
from load_imdb_data import load_imdb_data_into_df
from lstm_model import LSTMClassifier
from training_function import training_loop


def main():

    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    df = load_imdb_data_into_df()

    X = df['review']
    y = df['sentiment'].values

    #split for training and testing dataset (80/20)
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

    #split the training data further into training and validation dataset
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42)

    # info - shape of train data is 36000; shape of test data is 10000; shape of val data is 4000

    ##### encoding data

    x_train_encoded, x_val_encoded, x_test_encoded = encode_train_text_val_dataset(
                                                                                    x_train_data = x_train,
                                                                                    x_val_data = x_val,
                                                                                    x_test_data = x_test
                                                                                    )
    y_train = np.array([1 if y == "positive" else 0 for y in y_train])
    y_val = np.array([1 if y=="positive" else 0 for y in y_val])
    y_test = np.array([1 if y=="positive" else 0 for y in y_test])

    ### converting to a tensor

    X_train_tensor = torch.tensor(x_train_encoded, dtype=torch.long, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float, device=device)

    X_val_tensor = torch.tensor(x_val_encoded, dtype=torch.long, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float, device=device)

    X_test_tensor = torch.tensor(x_test_encoded, dtype=torch.long, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float, device=device)

    # TensorDataset wraps X and y tensors together so that indexing returns matched (input,label)
    train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor,y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor,y_test_tensor)

    # Creating batches using Dataloader (batch size = 32)

    batch_size = 32

    train_loader = DataLoader(train_dataset,batch_size,shuffle=True)

    val_loader = DataLoader(val_dataset,batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size, shuffle = True)

    #Initialising the LSTM classifier model 

    classifier_model = LSTMClassifier(vocab_size=20002,
                                    embed_dim=100,
                                    hidden_dim=128).to(device)

    #setup loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(),lr=0.001)

    #### training 
    training_start_time = time.perf_counter()
    trained_classifier_model = training_loop(
                  epochs=9, #epoch set to 9 train loss decreases but validation loss increases 
                  classifier_model=classifier_model, 
                  train_loader=train_loader,
                  val_loader = val_loader,
                  optimizer=optimizer,
                  loss_fn=loss_fn,
                  device=device,
                  path_trained_model= "sentiment_analysis/trained_models/lstm_epoch_9.pt",
                  seed=42
                  )
    training_end_time = time.perf_counter()
   
    print(f"time taken to run the training loop: {training_end_time - training_start_time}s")


if __name__ == main():
    main()