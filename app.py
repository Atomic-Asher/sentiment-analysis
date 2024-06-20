import streamlit as st
import pandas as pd
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to map sentiment labels to numerical values
def map_sentiment(sentiment):
    if sentiment == "POSITIVE":
        return 2
    elif sentiment == "NEGATIVE":
        return 0
    else:  # Assuming "NEUTRAL"
        return 1

# Function to process the uploaded file
def process_uploaded_file(file, file_type):
    # Convert the uploaded file to a DataFrame based on the file type
    if file_type == 'csv':
        df = pd.read_csv(file)
    elif file_type == 'xlsx':
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file type.")
        return None
    
    # Ensure all entries in the 'content' column are strings
    df['content'] = df['content'].astype(str)
    
    # Perform sentiment analysis on the 'content' column
    df['sentiment_label'] = df['content'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
    df['mapped_sentiment'] = df['sentiment_label'].apply(map_sentiment)
    
    return df

# Streamlit UI
st.title("Sentiment Analysis on Uploaded CSV or Excel File")

# Create a file uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Determine the file type
    file_type = uploaded_file.name.split('.')[-1]
    
    # Process the uploaded file
    df = process_uploaded_file(uploaded_file, file_type)
    
    if df is not None:
        # Display the results
        st.write("Results:")
        st.dataframe(df[['content', 'sentiment_label', 'mapped_sentiment']])
