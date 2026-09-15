import math
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


project_folder = Path(__file__).resolve().parents[1]
load_dotenv(project_folder / ".env")

client = OpenAI()

TOP_K = 10

st.title("Exercise 2.3: Ask My Document")

chunks_folder = project_folder / "chunks"

if not chunks_folder.exists():
    st.error("Cannot find the chunks folder. Please run Exercise 2.1 first.")
    st.stop()

chunk_files = sorted(chunks_folder.glob("*.txt"))

if not chunk_files:
    st.error("No .txt chunks found in the chunks folder.")
    st.stop()

chunk_texts = [file.read_text(encoding="utf-8") for file in chunk_files]

st.caption(f"{len(chunk_files)} chunks loaded")


def cosine_similarity(vector_1, vector_2):
    dot_product = sum(a * b for a, b in zip(vector_1, vector_2))
    length_1 = math.sqrt(sum(a * a for a in vector_1))
    length_2 = math.sqrt(sum(b * b for b in vector_2))
    return dot_product / (length_1 * length_2)


question = st.text_input("Ask a question about the document:")

if st.button("Ask document"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching the document..."):
                embedding_response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=[question] + chunk_texts,
                )

                question_embedding = embedding_response.data[0].embedding
                chunk_embeddings = [
                    item.embedding for item in embedding_response.data[1:]
                ]

                scores = [
                    cosine_similarity(question_embedding, chunk_embedding)
                    for chunk_embedding in chunk_embeddings
                ]

                ranked = sorted(
                    enumerate(scores), key=lambda pair: pair[1], reverse=True
                )[:TOP_K]

                excerpts = "\n\n".join(
                    f"[{chunk_files[idx].name}]\n{chunk_texts[idx]}"
                    for idx, _ in ranked
                )

                prompt = f"""
Answer the user's question using only the document excerpts below.

Question:
{question}

Document excerpts:
{excerpts}

Rules:
- Answer only using the excerpts above.
- Cite the file name after each claim, e.g. [chunk_007.txt].
- If the excerpts do not contain the answer, say:
  "I cannot find the answer in the retrieved document sections."
"""

                answer_response = client.responses.create(
                    model="gpt-4o",
                    input=prompt,
                )

            st.subheader("Answer")
            st.write(answer_response.output_text)

            st.subheader("Retrieved sources")
            for idx, score in ranked:
                label = f"{chunk_files[idx].name} — similarity {score:.3f}"
                with st.expander(label):
                    st.write(chunk_texts[idx])

        except Exception as error:
            st.error(f"OpenAI API error: {error}")