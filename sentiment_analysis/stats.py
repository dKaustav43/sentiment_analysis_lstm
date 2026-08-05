from collections import Counter
import numpy as np
import pandas as pd
from .vocab_and_encoding_functions import tokenize_text
from .loading_imdb_train_test_val_split import load_imdb_data_into_df, train_test_val_set
from typing import Annotated
from annotated_types import Interval

def fraction_word_coverage_distribution(texts:list[str]|pd.Series|np.ndarray) -> tuple[list[float],Counter]:
    """
    Compute the cumulative token-coverage distribution for a corpus of texts.

    Tokenizes each text, counts word frequencies across the whole corpus,
    then builds a cumulative distribution over words sorted by descending
    frequency. cumulative[i] is the fraction of all tokens accounted for
    by the top (i+1) most frequent words.

    Args:
        texts: Collection of raw text documents (e.g. reviews), as a list
            of strings, a pandas Series, or a numpy array.

    Returns:
        A tuple of:
            - cumulative_distribution: array where element i is the fraction
              of total tokens covered by the top (i+1) most frequent words.
            - counter: a Counter mapping each unique word to its frequency
              across the corpus.
    """ 
    
    tokenized_texts = tokenize_text(texts)
    all_tokens = []
    #first access the row (which is a single review), then loop over the single review appending each word to a list of all_tokens.
    for tokens in tokenized_texts:
        for words in tokens:
            all_tokens.append(words)
    
    # create a dictionary mapping for each unique word and how many times it appears
    counter = Counter(all_tokens)

    #sort frequencies descending
    freqs = np.array(sorted(counter.values(),reverse=True))
    total_tokens = freqs.sum()

    #cummulative coverage
    cumulative_distribution = np.cumsum(freqs)/total_tokens

    return cumulative_distribution, counter

def vocab_coverage_needed(cummulative_distribution:list[float], threshold:Annotated[float, Interval(ge=0,le=1)]):
    """
    Find how many top-frequency words are needed to reach a coverage threshold.

    Args:
        cumulative_distribution: Sorted, monotonically increasing array where
            element i is the fraction of total tokens covered by the top
            (i+1) most frequent words (as returned by
            fraction_word_coverage_distribution).
        threshold: Target coverage fraction, between 0.0 and 1.0 inclusive
            (e.g. 0.95 for 95% token coverage).

    Returns:
        The number of top-frequency words required to reach or exceed the
        given coverage threshold.
    """
    idx = np.searchsorted(cummulative_distribution,threshold) + 1
    return idx
    
def main():
    df_imdb = load_imdb_data_into_df()
    
    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_set(df=df_imdb)
    
    cumulative, counter  = fraction_word_coverage_distribution(texts = x_train) 

    print(f"Total number of unique words in the input texts is: {len(counter)}")

    threshold = 0.95
    idx = vocab_coverage_needed(cummulative_distribution=cumulative, threshold=0.95)
    print(f"words needed to cover {int(threshold*100)}% of tokens is {idx}")

if __name__ == "__main__":
    main()