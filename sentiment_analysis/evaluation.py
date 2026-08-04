from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

from lstm_model import LSTMClassifier
from load_imdb_data import load_imdb_data_into_df
from vocab_and_encoding import encode_train_text_val_dataset

# Evaluate one on the test set. Reporting on accuracy, Precision/Recall/F1.

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

    X_test_tensor = torch.tensor(x_test_encoded, dtype=torch.long, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float, device=device)

    # TensorDataset wraps X and y tensors together so that indexing returns matched (input,label)
    train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor,y_test_tensor)

    # Creating batches using Dataloader (batch size = 32)

    batch_size = 32

    train_loader = DataLoader(train_dataset,batch_size,shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size, shuffle = True)


    lstm_model_instance = LSTMClassifier(vocab_size=20002,
                                    embed_dim=100,
                                    hidden_dim=128).to(device)
    
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