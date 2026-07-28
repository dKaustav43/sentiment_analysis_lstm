from collections import Counter
import numpy as np
from pre_process_and_tokenize import tokenize_text
from sentiment_analysis import x_train

def vocab_coverage_stats(tokenized_texts):
    """
    Building a function with unique word coverage.
    """

    all_tokens = [words for tokens in tokenized_texts for words in tokens]
  
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

def main():
    train_tokens = tokenize_text(x_train)

    counter, cumulative = vocab_coverage_stats(tokenized_texts=train_tokens) # type: ignore

if __name__ == "__main__":
    main()