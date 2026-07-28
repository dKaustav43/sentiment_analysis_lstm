from collections import Counter

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

#encoding tokens - list[str] --> list[int]
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


#truncation (control sequence length)
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

#padding (make fixed length)
def pad_sequence(sequence, max_length, pad_value=0):
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
