import re

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