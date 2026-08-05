import torch.nn as nn

class LSTMClassifier(nn.Module):
    
    def __init__(self, vocab_size=20002, embed_dim=100, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.LSTM = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim,1) # 1-logit output is enough as it is binary cross-entropy. 

    def forward(self,x):
        x = self.embedding(x)
        x,(h,c) = self.LSTM(x) # x-output, h-hidden layer, c-final cell states.
        return self.fc(h[-1]) # h[-1]:last LSTM layers final hidden state.
    #the final hidden state encodes the sentiment-relevant information accumulated over the whole sequence.
