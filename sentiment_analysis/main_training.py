import numpy as np
import torch
import torch.nn as nn
import time

from loading_imdb_train_test_val_split import (load_imdb_data_into_df, 
                                              train_test_val_set, 
                                                 )
from vocab_and_encoding_functions import encode_train_text_val_dataset
from tensor_and_dataloader import converting_data_to_tensor, dataloader_batches
from lstm_model import LSTMClassifier
from training_function import training_loop


def main():

    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    df_imdb = load_imdb_data_into_df()
    
    # train-test-val split
    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_set(df=df_imdb)

    # encoding data
    x_train_encoded, x_val_encoded, x_test_encoded = encode_train_text_val_dataset(
                                                                                   x_train_data = x_train,
                                                                                    x_val_data = x_val,
                                                                                    x_test_data = x_test
                                                                                    )
    y_train_encoded = np.array([1 if y == "positive" else 0 for y in y_train])
    y_val_encoded = np.array([1 if y=="positive" else 0 for y in y_val])
    y_test_encoded = np.array([1 if y=="positive" else 0 for y in y_test])

    # converting encoded data to tensor
    X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = converting_data_to_tensor(
      x_train_encoded, x_val_encoded, x_test_encoded, y_train_encoded, y_val_encoded, y_test_encoded,
      device=device  
    ) 
    
    # dividing the tensors into batches ready for pytorch model
    train_loader, val_loader, test_loader = dataloader_batches(X_train_tensor, X_val_tensor, X_test_tensor, 
                                                               y_train_tensor, y_val_tensor, y_test_tensor,
                                                               batch_size=32)

    #Initialising the LSTM classifier model 

    classifier_model = LSTMClassifier().to(device)

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

    print("The trained classifier model state dict:")
    for param_tensor in trained_classifier_model.state_dict():
        print(param_tensor, "\t", trained_classifier_model.state_dict()[param_tensor].size())


if __name__ == main():
    main()