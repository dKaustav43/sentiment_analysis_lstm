import streamlit as st
import requests

st.title("🎬 Movie Review Sentiment Analysis")
user_review = st.text_area("Enter your movie review:")

if st.button("Run Inference"):
    
    if not user_review.strip():
        st.warning("⚠️ Please enter a movie review before submitting.")
    
    else:
        with st.spinner("Streaming text to FastAPI inference engine..."):
            response = requests.post(
                "http://localhost:8000/movie_reviews_sentiment/",
                json = {"movie_reviews": [user_review]}
                )
        
        if response.status_code == 200:
            st.success("Analysis Finished Successfully!")
            result = response.json()
            st.json(result)
        
        else:
            st.error(f"Backend Server Error ({response.status_code}): {response.text}")
