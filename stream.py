import streamlit as st
import numpy as np
import pandas as pd
import nltk 
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
st.title("Sentiment Analyser")
x = st.text_input("Enter your sentence:")
sia = SentimentIntensityAnalyzer()
if x:
    scores = sia.polarity_scores(x)
    st.write("Your sentiment analysis is:", scores)

    df = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative'],
        'Score': [scores['pos'], scores['neu'], scores['neg']]
    })

    fig, ax = plt.subplots()
    sns.barplot(data=df, x='Sentiment', y='Score', palette='pastel', ax=ax)
    ax.set_title("Sentiment Score Distribution", fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("")

    st.pyplot(fig)

#sia = SentimentIntensityAnalyzer()
#st.write("Sentiment Analyser")
#x = st.text_input("Enter your sentence:")
#y = sia.polarity_scores(x)
#st.write("Your sentiment analysis is:", y)