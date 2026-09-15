import streamlit as st
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

st.title("Exercise 2.2 Comparing Chunks")

# 1. Two text inputs
text_1 = st.text_area("Text 1", height=200)
text_2 = st.text_area("Text 2", height=200)


# 2. Create an embedding for each chunk
@st.cache_data(show_spinner=False)
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return response.data[0].embedding


# 3. Cosine similarity
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if st.button("Compare"):
    if not text_1.strip() or not text_2.strip():
        st.warning("Please enter both texts.")
    else:
        with st.spinner("Creating embeddings..."):
            try:
                emb_1 = get_embedding(text_1)
                emb_2 = get_embedding(text_2)
                score = cosine_similarity(emb_1, emb_2)
                st.metric("Cosine similarity", f"{score:.4f}")
                st.progress(max(0.0, min(1.0, score)))
                st.caption(f"Embedding dimension: {len(emb_1)}")
            except Exception as e:
                st.error(f"Failed: {e}")