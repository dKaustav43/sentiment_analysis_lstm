import torch
import torch.nn as nn
import time
from vocab_and_encoding_functions import imdb_vocab, encode_tokens, pad_sequence, truncation
from pre_process_and_tokenize import tokenize_text
from lstm_model import LSTMClassifier

def predict_sentiment_batch(
    texts: list[str], 
    classifier_model: nn.Module, 
    vocab = imdb_vocab(), 
    pad_length: int = 300, # set during the training

) -> list[dict]:
    """
    Takes a list of raw strings representing reviews, runs them through the trained model as a single batch,
    and returns a list of dictionaries containing the prediction results.
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    
    # Eval mode
    classifier_model.eval()
    
    tokens = tokenize_text(texts)
    token_ids = [encode_tokens(token,vocab) for token in tokens]
       
    # Pad with 0 or truncate to match the set max length
    if len(token_ids) < pad_length:
        token_ids = [pad_sequence(token, max_length=pad_length) for token in token_ids]
    else:
        token_ids = [truncation(token,max_length=pad_length) for token in token_ids]
               
    # 3. Convert the list of lists into a single 2D tensor: shape [batch_size, max_length]
    input_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    
    # 4. Forward pass through the model without calculating gradients
    with torch.no_grad():
        logits = classifier_model(input_tensor)
        
        # Flatten the logits if your model outputs an extra dimension like [batch_size, 1]
        if len(logits.shape) > 1 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
            
        # Apply sigmoid to convert raw logits into probabilities between 0.0 and 1.0
        probabilities = torch.sigmoid(logits).tolist()
        
    # 5. Format and construct the final output array
    results = []
    for text, prob in zip(texts, probabilities):
        # 0.50 threshold matches your validation loop logic
        label = "Positive" if prob > 0.50 else "Negative"
        confidence = prob if label == "Positive" else 1.0 - prob
        
        results.append({
            "text": text,
            "label": label,
            "confidence": round(confidence, 4)
        })
        
    return results

def main():
    test_reviews = ["The movie was a wonderful watch. I highly recommend!",
                    "I felt the movie was boring. Acting could have been better!",
                    "The movie was okay!"]
    
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    path_trained_model= "sentiment_analysis/trained_models/lstm_epoch_9.pt"

    lstm_model_instance = LSTMClassifier(vocab_size=20002,
                                    embed_dim=100,
                                    hidden_dim=128).to(device)
    
    lstm_model_instance.load_state_dict(torch.load(path_trained_model, weights_only=True))
    
    start_inference_time = time.perf_counter()
    results = predict_sentiment_batch(texts=test_reviews, classifier_model=lstm_model_instance)
    end_inference_time = time.perf_counter()
    print(f"total inference time is:{end_inference_time -start_inference_time}s")
    print(results)

if __name__ == "__main__":
    main()