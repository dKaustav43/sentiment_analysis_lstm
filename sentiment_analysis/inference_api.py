from fastapi import FastAPI
from .main_inference import predict_sentiment_batch
import torch
import time
from .lstm_model import LSTMClassifier
from pydantic import BaseModel
from datetime import datetime
from typing import Literal

device = torch.device("mps" if torch.mps.is_available() else "cpu")
path_trained_model= "sentiment_analysis/trained_models/lstm_epoch_9.pt"
lstm_model_instance = LSTMClassifier().to(device)
lstm_model_instance.load_state_dict(torch.load(path_trained_model, weights_only=True))

class ReviewsInput(BaseModel):
    movie_reviews: list[str]   

class ReviewSentiment(BaseModel):
    text: str
    label: Literal["Positive", "Negative"]
    confidence: float

class PublicOutput(BaseModel):
    Sentiment:list[ReviewSentiment]
    inference_time:str


app = FastAPI()

@app.post("/movie_reviews_sentiment/", response_model = PublicOutput)
def infer_movie_sentiment(payload:ReviewsInput):
    """
    Returns the sentiment of each movie review as "positive" or "negetive".
    Runs a trained LSTM classifier on IMDB dataset. 

    Input: Provide movie reviews as a list.
        Example: movie_reviews = [
                    "The movie was a wonderful watch. I highly recommend!",
                    "I felt the movie was boring. Acting could have been better!",
                    "The movie was okay!"
                    ]
    Output: 
    """
    start_inference_time = time.perf_counter()
    results = predict_sentiment_batch(texts=payload.movie_reviews,
                                     classifier_model=lstm_model_instance,
                                     device=device)
    end_inference_time = time.perf_counter()
    total_inference_time = f"{end_inference_time - start_inference_time}s"

    movie_sentiment = {}
    movie_sentiment["Sentiment"] = results
    movie_sentiment["inference_time"] = total_inference_time

    return movie_sentiment

