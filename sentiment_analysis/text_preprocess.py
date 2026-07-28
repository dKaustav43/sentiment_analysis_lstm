import re

def preprocess_string(text:str):
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


def tokenize_text(texts:list[str]) -> list[list[str]]:
    """
    Preprocess and tokenize a collection of IMDb reviews.
    Eac review is cleaned using 'preprocess_string()
    and then split into a list of white-space separated tokens. 
    """
    return [preprocess_string(text).split() for text in texts]