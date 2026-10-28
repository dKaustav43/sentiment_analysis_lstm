import time
from pathlib import Path
from typing import Literal, Protocol

import boto3
import torch
from botocore.exceptions import ClientError
from fastapi import FastAPI
from pydantic import BaseModel

from .lstm_model import LSTMClassifier
from .main_inference import predict_sentiment_batch

# add a lifespan here. 


## the device is CPU for local dev and AWS instance for the new deployment. 
## use map_location to remap the present device. Necessary as the device on which the model is trained maynot be the device where the inference is running.
path_trained_model= "sentiment_analysis/trained_models/lstm_epoch_9.pt"
device = torch.device("mps" if torch.mps.is_available() else "cpu")
state_dict = torch.load(path_trained_model, map_location=device, weights_only=True)

## make this such that multiple models can be loaded.
lstm_model_instance = LSTMClassifier().to(device)
lstm_model_instance.load_state_dict(state_dict)

## two paths - one local for prod and the other S3 bucket to store the model.
class ModelStore(Protocol):
    """
    Interface to load the model from different platforms. 
    """
    def load_bytes(self) -> bytes: ...

class LocalModelStore:
    """
    Loads model file stored as .pt locally into memory.
    This is for local development purposes.
    """
    def __init__(self, path:str):
        self.path = Path(path)
    
    def load_bytes(self) -> bytes:
        return self.path.read_bytes()

class S3ModelStore:
    """
    Loads the model file in memory from S3 bucket.
    This is for production to AWS.
    """
    def __init__(self, bucket:str, key:str):
        self.bucket = bucket
        self.key = key
    
    def load_bytes(self) -> bytes:
        try:
            client = boto3.client('s3')
            obj = client.get_object(Bucket = self.bucket, Key = self.key)
            return obj['Body'].read()
        
        except ClientError as e:
            if e.response['Error']['Code'] == "NoSuchKey":
                print(f"Key {self.key} not found")
            raise            

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

@app.get("/", include_in_schema=False)
def health():
    return {"status": "ok"}

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

