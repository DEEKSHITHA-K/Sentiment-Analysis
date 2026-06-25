import streamlit as st
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize analyzers
sia = SentimentIntensityAnalyzer()
bert_pipeline = pipeline("sentiment-analysis")

# Function for VADER sentiment
def vader_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] > 0.05:
        return "Positive"
    elif score['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function for BERT sentiment
def bert_sentiment(text):
    result = bert_pipeline(text)[0]
    return f"{result['label']} ({result['score']:.2f})"

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="wide")

st.title("📊 Sentiment Analysis Project")
st.write("Analyze text sentiment using **Rule-based (VADER)** and **Deep Learning (BERT)**.")

# Text input
st.subheader("🔹 Analyze Single Text")
user_text = st.text_area("Enter text here:", "I love this product!")

if st.button("Analyze Sentiment"):
    st.write("**VADER Result:**", vader_sentiment(user_text))
    st.write("**BERT Result:**", bert_sentiment(user_text))

# File upload for batch processing
st.subheader("🔹 Analyze Sentiments from a File (CSV)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must contain a column named 'text'")
    else:
        df["VADER Sentiment"] = df["text"].apply(vader_sentiment)
        df["BERT Sentiment"] = df["text"].apply(bert_sentiment)
        st.dataframe(df)
        st.download_button("Download Results", df.to_csv(index=False), "sentiment_results.csv", "text/csv")
