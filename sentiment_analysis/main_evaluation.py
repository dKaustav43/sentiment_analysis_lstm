import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from lstm_model import LSTMClassifier
from loading_imdb_train_test_val_split import load_imdb_data_into_df, train_test_val_set
from vocab_and_encoding_functions import encode_train_text_val_dataset
from tensor_and_dataloader import converting_data_to_tensor, dataloader_batches
                                               
def eval_on_test_set(
        trained_classifier_model:nn.Module,
        test_loader:DataLoader,
        train_loader:DataLoader,
        loss_function
):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    trained_classifier_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0 

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = trained_classifier_model(X_batch).squeeze(1)
            loss = loss_function(logits,y_batch)
            test_loss += loss.item()

            preds = (torch.sigmoid(logits)>0.50).long()
            correct += (preds == y_batch.long()).sum().item()
            total += y_batch.size(0)

        print(
            f"Test loss: {test_loss/len(train_loader):.4f} | "
            f"Test Acc: {correct/total:.4f}"
        )   

def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    path_trained_model= "sentiment_analysis/trained_models/lstm_epoch_9.pt"

    df_imdb = load_imdb_data_into_df()

    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_set(df=df_imdb)

    x_train_encoded, x_val_encoded, x_test_encoded = encode_train_text_val_dataset(x_train, x_val, x_test)
    
    y_train_encoded = np.array([1 if y == "positive" else 0 for y in y_train])
    y_val_encoded = np.array([1 if y=="positive" else 0 for y in y_val])
    y_test_encoded = np.array([1 if y=="positive" else 0 for y in y_test])

    X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = converting_data_to_tensor(
      x_train_encoded, x_val_encoded, x_test_encoded, y_train_encoded, y_val_encoded, y_test_encoded,
      device=device  
    ) 
    train_loader, val_loader, test_loader = dataloader_batches(X_train_tensor, X_val_tensor, X_test_tensor, 
                                                               y_train_tensor, y_val_tensor, y_test_tensor,
                                                               batch_size=32)


    lstm_model_instance = LSTMClassifier().to(device)
    
    lstm_model_instance.load_state_dict(torch.load(path_trained_model, weights_only=True))

    loss_fn = nn.BCEWithLogitsLoss()

    start_eval = time.perf_counter()
    eval_on_test_set(
        trained_classifier_model = lstm_model_instance,
        test_loader=test_loader,
        train_loader=train_loader,
        loss_function=loss_fn,
    )
    end_eval = time.perf_counter()

    print(f"Time taken to run the eval loop: {end_eval - start_eval}s")

if __name__ == "__main__":
    main()