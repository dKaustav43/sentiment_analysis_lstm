import torch
from torch.utils.data import TensorDataset, DataLoader

def converting_data_to_tensor(x_train_encoded, x_val_encoded, x_test_encoded, 
                              y_train_encoded, y_val_encoded, y_test_encoded, device
                              ):
    """
    Custom function to convert IMDB data into tensors.
    The data must be split into train/test/val and encoded using xxxxx
    prior to imputing them into this function.
    """
    
    X_train_tensor = torch.tensor(x_train_encoded, dtype=torch.long, device=device)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.float, device=device)

    X_val_tensor = torch.tensor(x_val_encoded, dtype=torch.long, device=device)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.float, device=device)

    X_test_tensor = torch.tensor(x_test_encoded, dtype=torch.long, device=device)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.float, device=device)

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

def dataloader_batches(x_train_tensor,x_val_tensor,x_test_tensor, 
                       y_train_tensor, y_val_tensor, y_test_tensor,
                        batch_size:int = 32):
    """
    The imdb tensors is converted into batches of 32 and prepared using Dataloader to be fed into the 
    deep learning model.
    This is a custom step applied to dataset generated 
    using the function above (converting_data_to_tensor) as input args. 
    """

    train_dataset = TensorDataset(x_train_tensor,y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor,y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor,y_test_tensor)

    train_loader = DataLoader(train_dataset,batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle = True)

    return train_loader, val_loader, test_loader