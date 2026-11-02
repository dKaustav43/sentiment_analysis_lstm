import httpx
from fastapi.testclient import TestClient

from sentiment_analysis.inference_api import app

client = TestClient(app)

def test_health_endpoint():
    """Testing the endpoint in app. No need to run the server.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status" : "ok"}

def test_server_running():
    """Testing the endpoint over a server. Needs the server to run.
    """
    try:
        response = httpx.get("http://0.0.0.0:8000")
    except httpx.ConnectError:
        print("Server is not running on port 8000")
        return
    
    assert response.status_code == 200
    assert response.json() == {"status" : "ok"}
        
def test_infer_movie_sentiment_over_server():
    """Testing the inference  endpoint when the server is running. 
    If server not running, then it returns a connection error and passes the test.
    """
    payload = {"movie_reviews": ["This movie is great!"]}
    try:
        r = httpx.post("http://0.0.0.0:8000/movie_reviews_sentiment/", json = payload)
    except httpx.ConnectError:
        print("Server is not running on port 8000")
        return
    
    assert r.status_code == 200

    response = r.json()

    assert "Sentiment" in response
    assert "inference_time" in response

    sentiment = response["Sentiment"][0]

    assert sentiment["label"] in ["Positive", "Negative"]

    assert 0 <= sentiment["confidence"] <= 1

def test_infer_movie_sentiment():
    """Tests the sentiment analysis endpoint in app. 
       This is configured to run the lifespan event and it is not required to start the server.     
    """
    with TestClient(app) as client:
        r = client.post("/movie_reviews_sentiment/",
                            json = {"movie_reviews" :["The movie was great!"]})
        assert r.status_code == 200

        response = r.json()

        assert "Sentiment" in response
        assert "inference_time" in response

        sentiment = response["Sentiment"][0]

        assert sentiment["label"] in ["Positive", "Negative"]

        assert 0 <= sentiment["confidence"] <= 1


#test 3: Test the lifespan event