import streamlit as st
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gtts import gTTS

# Load models
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
kw_model = KeyBERT(model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")

# Streamlit interface
st.title("Text Analysis Tool")
st.write("Enter a paragraph below and choose the analysis you'd like to perform.")

# Input text
text = st.text_area("Enter paragraph here", height=200)

if text:
    # Display buttons for different features
    if st.button("Find Keywords"):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', use_mmr=True, diversity=0.3)
        st.write("### Extracted Keywords:")
        for keyword in keywords:
            st.write(f"- {keyword[0]} (Score: {keyword[1]:.4f})")

    if st.button("Summarize Text"):
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        st.write("### Summary:")
        st.write(summary[0]['summary_text'])

    if st.button("Analyze Sentiment"):
        sentiment_model = pipeline("sentiment-analysis")
        sentiment = sentiment_model(text)
        st.write("### Sentiment Analysis:")
        st.write(f"**{sentiment[0]['label']}** with a score of **{sentiment[0]['score']:.4f}**")

    if st.button("Highlight Named Entities"):
        doc = nlp(text)
        st.write("### Named Entities:")
        for ent in doc.ents:
            st.write(f"- **{ent.text}** ({ent.label_})")

    if st.button("Generate Word Cloud"):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    if st.button("Text to Speech"):
        tts = gTTS(text)
        tts.save("output.mp3")
        audio_file = open("output.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")

# Streamlit styles for beautification
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)
