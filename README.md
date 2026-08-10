# Sentiment Analysis with LSTM

LSTM model for sentiment analysis on IMDB dataset.

The data `imdb_dataset.csv` is taken from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews.
The data is in the data/ folder.

To have a look at the dataset, basic Data and vocab characteristics, encoding and tensor generation - go to the folder **sentiment_analysis/quarto_reports** and open **sentiment_analysis.html** in your web browser. 

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

### 4. Run using Streamlit frontend

A simple streamlit frontend (sentiment_analysis/streamlit_frontend.py)is created to get the sentiment out of a movie review using a simple and nice UI. Before running streamlit, make sure the fastapi server is running in the backend (step3).
Run streamlit app using the following command -

```zsh
streamlit run sentiment_analysis/streamlit_frontend.py
```
Now you can enter a movie review in the box and click on run inference. 

## Run inference inside Docker container

I have provided a docker file here. It installs uv, uses uv to install all the dependencies and we will use uv run to run the program. 

Docker image needs to be build here. I have not shared the image on the docker hub. 

1. Build the Docker image

```sh
docker build -t movie-sentiment-analysis:1.0 .
```
check the image is built successfully using 
```sh
docker image ls
```

2. Run the Docker container

```sh
docker run -d -p 8000:8000 --name movie-review-sentiment-analysis movie-sentiment-analysis:1.0
```
check that the container is running using the following command in your terminal

```sh
docker ps
```
3. Run the fastapi endpoint using UV from inside the container. In your terminal, run the following command - 

```sh
docker exec -it movie-review-sentiment-analysis uv run fastapi run sentiment_analysis/inference_api.py --host 0.0.0.0 --port 8000
```
now, go to the brower and run **localhost:8000/docs** and you will see the fastapi docs where you can execute the inference and try it out just as described above.

4. To debug or run other parts of the application from inside the container, spin up a bash terminal from within the container using the following command -

```sh 
docker exec -it movie-review-sentiment-analysis sh
```

**Note** - I have only included the Docker commands to spin up a fastapi server. For streamlit frontend, I will be making a docker compose file shortly. For now, the option of running streamlit frontend is not possible. 
