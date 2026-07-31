import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .vocab_and_encoding import encode_train_text_val_dataset
from .vocab_and_encoding import build_vocab, encode_tokens, truncation, pad_sequence
from .load_imdb_data import load_imdb_data_into_df


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

    # # Building the Model - Basic building blocks for an LSTM model. 

    import torch.nn as nn
    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.LSTM = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim,1) # 1-logit output is enough as it is binary cross-entropy. 

        def forward(self,x):
            x = self.embedding(x)
            x,(h,c) = self.LSTM(x) # x-output, h-hidden layer, c-final cell states.
            return self.fc(h[-1]) # h[-1]:last LSTM layers final hidden state.
        #the final hidden state encodes the sentiment-relevant information accumulated over the whole sequence.

    classifier_model = LSTMClassifier(vocab_size=20002,
                                    embed_dim=100,
                                    hidden_dim=128).to(device)
    print(classifier_model)

    #setup loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(),lr=0.001)

    #training loop

    torch.manual_seed(42)
    epochs = 10


    for epoch in range(epochs):

        #training 
        classifier_model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = classifier_model(X_batch).squeeze(1)
            loss = loss_fn(logits,y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        #validation 
        classifier_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0 

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = classifier_model(X_batch).squeeze(1)
                loss = loss_fn(logits,y_batch)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits)>0.50).long()
                correct += (preds == y_batch.long()).sum().item()
                total += y_batch.size(0)

        print(
            f"Epoch {epoch+1} | "
            f"Train loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} |"
            f"Val Acc: {correct/total:.4f}"
        )   


    # Around 10 Epochs are enough for the training. 

    # # Final Evaluation

    # Evaluate one on the test set. Reporting on accuracy, Precision/Recall/F1.

    #validation 
    classifier_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0 

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = classifier_model(X_batch).squeeze(1)
            loss = loss_fn(logits,y_batch)
            test_loss += loss.item()

            preds = (torch.sigmoid(logits)>0.50).long()
            correct += (preds == y_batch.long()).sum().item()
            total += y_batch.size(0)

        print(
            f"Test loss: {test_loss/len(train_loader):.4f} | "
            f"Test Acc: {correct/total:.4f}"
        )   



    vocab = build_vocab(train_tokens,max_vocab_size=20000)


    # Saving the model
    # classifier_model = LSTMClassifier(vocab_size=20002,  # +2 for PAD and UNK
    #                                   embed_dim=100,
    #                                   hidden_dim=128).to(device)



    #Inference function/layer.


    def predict_sentiment(
            text:str,
            vocab=vocab,
            max_length=300,
            device=device
    ):
        classifier_model.eval()
        #tokenize
        tokens = text.lower().split()

        #encode
        encoded = encode_tokens(tokens, vocab)
        encoded = truncation(encoded, max_length)
        encoded = pad_sequence(encoded, max_length)


        x = torch.tensor(encoded).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = classifier_model(x).squeeze()
            prob = torch.sigmoid(logit).item()

        label = "positive" if prob >= 0.5 else "negetive"

        return label, prob


    label,prob = predict_sentiment(
        "The movie was was well done!"
    )
    print(label,prob)


if __name__ == main():
    main()