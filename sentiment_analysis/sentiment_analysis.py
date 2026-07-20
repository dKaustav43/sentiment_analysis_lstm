import pandas as pd
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch

#update device fallback - cuda --> MPS ---> CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#loading the imdb dataset - add a function
base_csv = 'data/IMDB Dataset.csv'
df = pd.read_csv(base_csv)

#assigning the X,y values and splitting training and testing set - split the training or the test dataset 
X,y = df['review'].values, df['sentiment'].values

#split for training and testing dataset (80/20)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

#split the training data further into testing, training and validation dataset
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

print(f'shape of train data is {x_train.shape}')
print(f'shape of test data is {x_test.shape}')
print(f'shape of val data is {x_val.shape}')


# # Data Pre-processing - NLP knowledge kicks in
# 
# 1. Clean text using preprocess_string() function. 
# 2. Split into tokens.
# 3. Build a vocabulary by reserving tokens for <PAD> - Padding and
#     <UNK> - unknown words.
# 4. Convert to Pytorch Tensors
# 
# Typical hyperparameters for IMDB
# max_vocab_size = 20000
# max_len = 200

def preprocess_string(text:str):
    #lowercasing to reduce vocabulary size
    text = text.lower()
    #normalize punctuation - keep !,? for sentiment.
    text = re.sub(r"[^a-z0-9!?']", " ", text)
    #normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    #remove html tags
    text = re.sub(r"<.*?>", "", text) 

    return text    


# Checking the token length charecteristics of reviews
rev_token_length = [len(preprocess_string(text).split()) for text in X]
pd.Series(rev_token_length).describe()

def tokenize_text(texts):
    return [preprocess_string(text).split() for text in texts]

counter = Counter()
tokenized_text = tokenize_text(x_train)
for token in tokenized_text:
    counter.update(token)
    
#vocabulary building with top 20000 unique words
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def build_vocab(tokenized_texts, max_vocab_size = 20000):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    vocab = {
        word:idx+2 
        for idx, (word,_) in enumerate(counter.most_common(max_vocab_size))
    }
    vocab[PAD_TOKEN] = 0
    vocab[UNK_TOKEN] = 1
    return vocab

#encoding tokens - list[str] --> list[int]
def encode_tokens(tokens,vocab):
    unk_idx = vocab[UNK_TOKEN]
    return [vocab.get(word,unk_idx) for word in tokens]


#truncation (control sequence length)
def truncation(sequence, max_length):
    return sequence[:max_length]

#padding (make fixed length)
def pad_sequence(sequence, max_length, pad_value=0):
    return sequence + [pad_value] * (max_length - len(sequence))


# # Full dataset encoding

def prepare_dataset(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        max_vocab_size = 20000,
        max_length = 300
):
    #tokenize
    train_tokens = tokenize_text(x_train)
    val_tokens = tokenize_text(x_val)
    test_tokens = tokenize_text(x_test)

    #vocab
    vocab = build_vocab(train_tokens,max_vocab_size)

    #encode
    X_train_encoded = [
        pad_sequence(
            truncation(encode_tokens(tokens,vocab),max_length),
            max_length
            )
            for tokens in train_tokens
    ]
    X_val_ecoded = [
        pad_sequence(
            truncation(encode_tokens(tokens,vocab),max_length),
            max_length
            )
            for tokens in val_tokens
    ]
    x_test_encoded = [
        pad_sequence(
            truncation(encode_tokens(tokens,vocab),max_length),
            max_length
            )
            for tokens in test_tokens
    ]
    #labels 
    y_train = np.array([1 if y == "positive" else 0 for y in y_train])
    y_val = np.array([1 if y=="positive" else 0 for y in y_val])
    y_test = np.array([1 if y=="positive" else 0 for y in y_test])

    return (
        np.array(X_train_encoded),
        y_train,
        np.array(X_val_ecoded),
        y_val,
        np.array(x_test_encoded),
        y_test
    )

x_train_encoded, y_train, x_val_encoded, y_val, x_test_encoded, y_test = prepare_dataset(
    x_train,y_train,x_val,y_val, x_test, y_test
)


# Building a function to check the unique word coverage

def vocab_coverage_stats(tokenized_texts):

    all_tokens = [words for tokens in tokenized_texts for words in tokens]

    #count the frequency    
    counter = Counter(all_tokens)

    #sort frequencies descending
    freqs = np.array(sorted(counter.values(),reverse=True))
    total_tokens = freqs.sum()

    #cummulative coverage
    cumulative = np.cumsum(freqs)/total_tokens

    threshold = [0.90,0.95,0.98]
    coverage_words = {}

    for t in threshold:
        idx = np.searchsorted(cumulative,t) + 1
        coverage_words[t] = idx

    print("total unique words in the training set:",len(counter))
    for t,words_needed in coverage_words.items():
        print(f"Words needed to cover {int(t*100)}% of tokens:{words_needed}")

    return counter, cumulative


train_tokens = tokenize_text(x_train)

counter, cumulative = vocab_coverage_stats(train_tokens)

# So a max_vocabulary size of 20,000 covers more than 95% of tokens.
# 
# The rest is very rare words typos etc. The rare words get very few gradient updates. Their embeddings stay poorly trained. Model capacity is wasted on noise. They contribute very little to sentiment. 
# 
# Transformers don't use this approach. they usually do subword tokenization (BPE, WordPiece).

# Convert Numpy arrays into Torch Tensors - Pytorch basics and RNN architecture - Why RNN? Why better than simple Neural networks? 
#  How do transformers improve on this? 

X_train_tensor = torch.tensor(x_train_encoded, dtype=torch.long, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float, device=device)

X_val_tensor = torch.tensor(x_val_encoded, dtype=torch.long, device=device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float, device=device)

X_test_tensor = torch.tensor(x_test_encoded, dtype=torch.long, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float, device=device)


train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
val_dataset = TensorDataset(X_val_tensor,y_val_tensor)
test_dataset = TensorDataset(X_test_tensor,y_test_tensor)


# Creating batches using Dataloader (batch size = 32)

batch_size = 32

train_loader = DataLoader(train_dataset,batch_size,shuffle=True)

val_loader = DataLoader(val_dataset,batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size, shuffle = True)

xb , yb = next(iter(train_loader))
print(xb.shape)
print(yb.shape)
print(xb.dtype)


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