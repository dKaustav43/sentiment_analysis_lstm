# Sentiment Analysis with LSTM

LSTM model for sentiment analysis on IMDB dataset.

The data `imdb_dataset.csv` is taken from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews.
The data is in the data/ folder.

## Project Setup and Execution Guide

This guide explains how to clone the repository, synchronize the environment,
and run the project scripts.

## Repository

GitHub Repository:

https://github.com/dKaustav43/sentiment_analysis_lstm

## Prerequisites

Ensure that **Git** and **uv** are installed on your system.

To install uv, check the following docs: 
https://docs.astral.sh/uv/getting-started/installation

## Getting Started

Follow the steps below in Terminal:

### 1. Clone the repository

```zsh
git clone https://github.com/dKaustav43/sentiment_analysis_lstm.git
cd sentiment-analysis-lstm
```

### 2. Synchronize the Environment

This repository includes a pyproject.toml and uv.lock file. Running the following command will automatically:

 - Create the virtual environment (.venv)
 - Install the exact dependency versions defined in the lock file

```zsh
uv sync
```

### 3. Run Inference

```zsh
fastapi run sentiment_analysis/inference_api.py 
```
open the FastAPI docs in your browser at  

localhost:8000/docs

Try out the POST/moview_reviews_sentiment/ 

In the parameter section, under Request body, you send a request, such as:

```json
{
    "movie_reviews" : [
        "I loved the movie!",
        "The movie was boring. I didn't like the movie much"

    ]
        
}

```
Click the execute command, which executes inference using the trained LSTM model. 
The output of the inference will be seen in the Response Body such as:

```json
{
  "Sentiment": [
    {
      "text": "I loved the movie!",
      "label": "Positive",
      "confidence": 0.8385
    },
    {
      "text": "The movie was boring. I didn't like the plot much",
      "label": "Negative",
      "confidence": 0.9454
    }
  ],
  "inference_time": "0.13856225000927225s"
}

```



