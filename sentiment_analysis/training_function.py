import torch
import torch.nn as nn
from torch.utils.data import DataLoader
## training loop
def training_loop(epochs:int, 
                  classifier_model:nn.Module, 
                  train_loader:DataLoader,
                  val_loader:DataLoader,
                  optimizer,
                  loss_fn,
                  device,
                  seed:int = 42):
    
    torch.manual_seed(seed)

    for epoch in range(epochs):
 
        classifier_model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = classifier_model(X_batch).squeeze(1) #squeezing the dimension to match shape of the labels
            loss = loss_fn(logits,y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), max_norm=1.0) #limits the maximum value of gradient to 1.0 and hence preventing vanishing gradients.
            optimizer.step() #update the model weights and biases.

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

                preds = (torch.sigmoid(logits)>0.50).long() # .long() converts dtype to a 64-bit integer hence forcing prob<0.50 to 0 and prob>0.50 to 1.
                correct += (preds == y_batch.long()).sum().item() # adding the correct predictions
                total += y_batch.size(0) # adding the total size of each batch

        print(
            f"Epoch {epoch+1} | "
            f"Train loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} |"
            f"Val Acc: {correct/total:.4f}"
        )   
