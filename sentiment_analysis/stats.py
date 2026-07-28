from collections import Counter
import numpy as np
import pandas as pd
from pre_process_and_tokenize import tokenize_text
from load_imdb_data import load_imdb_data_into_df, train_test_val_set
from typing import Annotated
from annotated_types import Interval

def fraction_word_coverage_distribution(texts:list[str]|pd.Series|np.ndarray) -> tuple[list[float],dict]:
    """
    cummulative[i] is fraction of all tokens that are accounted for by the top i+1 words. 
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