import streamlit as st
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# Load the MiniLM model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize KeyBERT with the loaded model
kw_model = KeyBERT(model)

# Streamlit interface
st.title("Keyword Extractor")
st.write("Enter a paragraph below and click 'Find Keywords' to extract keywords based on context.")

# Input text
text = st.text_area("Enter paragraph here", height=200)

# Button to find keywords
if st.button("Find Keywords"):
    if text:
        # Extract keywords
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', use_mmr=True, diversity=0.2)
        
        # Display keywords
        st.write("### Extracted Keywords:")
        for keyword in keywords:
            st.write(f"- {keyword[0]} (Score: {keyword[1]:.4f})")
    else:
        st.write("Please enter a paragraph to extract keywords.")

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
