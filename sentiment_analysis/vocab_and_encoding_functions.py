import re
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from collections import Counter
from sentiment_analysis.loading_imdb_train_test_val_split import load_imdb_data_into_df, train_test_val_set

##### Pre-processing text and tokenisation utils #####

def preprocess_string(text:str) -> str:
    """
    A basic script to clean and normalize texts.
    """
    #lowercasing to reduce vocabulary size
    text = text.lower()
    #normalize punctuation - keep !,? for sentiment.
    text = re.sub(r"[^a-z0-9!?']", " ", text)
    #normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    #remove html tags
    text = re.sub(r"<.*?>", "", text) 

    return text


def tokenize_text(texts:list[str]| pd.Series[str]| NDArray[np.str_]) -> list[list[str]]:
    """
    Preprocess and tokenize a collection of IMDb reviews.
    Eac review is cleaned using 'preprocess_string()
    and then split into a list of white-space separated tokens. 
    """
    return [preprocess_string(text).split() for text in texts]

##### Building custom IBMD vocabulary util #####

#vocabulary building with top 20000 unique words
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def build_vocab(tokenized_texts:list[list[str]], max_vocab_size = 20000) -> dict[str,int]:
    """ 
    Building the vocab includes the following steps:
    1. Counter is used to provide a dictionary of word frequencies.
    2. Vocab is a dictionary of top 20,000 most common words in IMDb review set with corresponding ID's.
        Note: ID = 1 is reserved for <PAD> token.
              ID = 2 is reserved for <UNK> token.

    *Note* Using only top 20000 words sacrifices the ability to distinguish rare words, but usually has little 
    impact on sentiment analysis as sentiment can be expressed in common wors - good, boring, love, excellent, funny etc.

    Keeping fewer parameters enables 1. training faster 2. uses less memory 3. reduces overfitting .

    LLM's would be better at this task as they use subword tokenisation and they are usually pretrained on huge datasets. 

    Output:
        Vocab:dict 
        example: {"PAD_TOKEN" : 0,
                "UNK_TOKEN" : 1,
                "The" : 3,
                ...}
    """
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

##### The IMDB vocabulary #### 

def imdb_vocab() -> dict[str,int]:
    """ (This will be seaparated out into a script later.)
    This standalone function builds the vocabulary from the training data of IMDB dataset.
    This will be used by other functions for example to encode tokens before training and inference.

    Output:
        Vocab:dict 
        example: {"PAD_TOKEN" : 0,
                "UNK_TOKEN" : 1,
                "The" : 3,
                ...}
    """
    
    df_imdb = load_imdb_data_into_df()
    # train-test-val split
    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_set(df = df_imdb)
    x_training_imdb_tokenized = tokenize_text(x_train)
    vocab_imdb = build_vocab(tokenized_texts=x_training_imdb_tokenized)

    return vocab_imdb


##### Encoding Word Tokens Utils #####
def encode_tokens(tokens:list[str],vocab:dict[str,int]) -> list[int]:
    """
    Encode a sequence of tokens into their corresponding integer indices.

    Each token is mapped to its index in the vocabulary. Tokens that are
    not present in the vocabulary are assigned the index of the `<UNK>`
    (unknown) token.

    Args:
        tokens (list[str]): A list of tokenised words.
        vocab (dict[str, int]): A mapping from words to integer indices.

    Returns:
        list[int]: A list of integer token IDs.

    Example:
        vocab = {"<PAD>": 0, "<UNK>": 1, "movie": 2, "good": 3}
        encode_tokens(["good", "movie", "excellent"], vocab)
        [3, 2, 1]
    
    """
    unk_idx = vocab[UNK_TOKEN]
    return [vocab.get(word,unk_idx) for word in tokens]

def truncation(sequence:list[int], max_length:int) -> list[int]:
    """
    Truncate a sequence to a maximum length.

    If the sequence exceeds `max_length`, only the first `max_length`
    elements are retained. Shorter sequences are returned unchanged.

    Args:
        sequence (list[int]): A sequence of encoded token IDs.
        max_length (int): The maximum allowed sequence length.

    Returns:
        list[int]: The truncated sequence.

    Example:
        truncation([5, 3, 4, 2, 8, 9], 4)
        [5, 3, 4, 2]
    """
    return sequence[:max_length]

def pad_sequence(sequence:list[int], max_length, pad_value=0) -> list[int]:
    """
    Pad a sequence to a fixed length.

    Padding values are appended to the end of the sequence until its
    length equals `max_length`. The padding value typically corresponds
    to the `<PAD>` token in the vocabulary.

    Args:
        sequence (list[int]): A sequence of encoded token IDs.
        max_length (int): The desired sequence length.
        pad_value (int, optional): The value used for padding.
            Defaults to 0.

    Returns:
        list[int]: The padded sequence.

    Example:
        pad_sequence([5, 3, 4], 6)
        [5, 3, 4, 0, 0, 0]
    """
    return sequence + [pad_value] * (max_length - len(sequence))

##### Custom function to encode IMDB dataset before Tensorisation #####
# Full dataset encoding function
def encode_train_text_val_dataset(
        x_train_data:list[str],
        x_val_data:list[str],
        x_test_data:list[str],
        max_length = 300
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """
    This function tokenizes training, validation and testing datasets and then encodes them to uniform length sequences.  
    """
    #tokenize
    train_tokens = tokenize_text(x_train_data)
    val_tokens = tokenize_text(x_val_data)
    test_tokens = tokenize_text(x_test_data)

    #vocab
    vocab = imdb_vocab()

    #encode
    X_train_encoded = [
        pad_sequence(
            truncation(encode_tokens(tokens,vocab),max_length),
            max_length
            )
            for tokens in train_tokens
    ]
    X_val_encoded = [
        pad_sequence(
            truncation(encode_tokens(tokens,vocab),max_length),
            max_length
            )
            for tokens in val_tokens
    ]
    X_test_encoded = [
        pad_sequence(
            truncation(encode_tokens(tokens,vocab),max_length),
            max_length
            )
            for tokens in test_tokens
    ]

    return X_train_encoded, X_val_encoded, X_test_encoded