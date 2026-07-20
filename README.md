# Sentiment Analysis with LSTM

LSTM model for sentiment analysis on IMDB dataset.

## Prerequisites

- Docker Desktop installed and running
- Git installed

## Quick Start

### 1. Clone the repository

```zsh
git clone https://github.com/dKaustav43/sentiment_analysis_lstm.git
cd sentiment-analysis-lstm
```

### 2. Add Your Dataset

Place `imdb_dataset.csv` from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews in the project root.

### 3. build Docker image
docker build -t sentiment-analysis .

### 4. Run the container
```bash
docker run --rm -p 8888:8888 -v "$(pwd)":/home/jovyan/work sentiment-analysis:v1
```

**On Windows, use:**
```bash
docker run --rm -p 8888:8888 -v "%cd%":/home/jovyan/work sentiment-analysis:v1
```

### 5. Access jupyter notebook
- Open your browser to `http://localhost:8888`
- Copy the token from the terminal output
- Open `sentiment_analysis.ipynb`

## Stopping the Container

Press `Ctrl+C` in the terminal where the container is running.


Access Jupyter at http://localhost:8888

## Notes

- The CSV file is not included in the repository (too large)
- All changes to the notebook are saved to your local machine
- Container is automatically removed after stopping (due to `--rm` flag)