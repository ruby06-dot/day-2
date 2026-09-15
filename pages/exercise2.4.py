from pathlib import Path

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

project_folder = Path(__file__).resolve().parents[1]
load_dotenv(project_folder / ".env")

import os

client = OpenAI()

TOP_K = 10
DB_PATH = str(project_folder / "chroma_db")
COLLECTION_NAME = "document_chunks"

st.title("Exercise 2.4: Ask My Document (Chroma)")

chunks_folder = project_folder / "chunks"

if not chunks_folder.exists():
    st.error("Cannot find the chunks folder. Please run Exercise 2.1 first.")
    st.stop()

chunk_files = sorted(chunks_folder.glob("*.txt"))

if not chunk_files:
    st.error("No .txt chunks found in the chunks folder.")
    st.stop()


@st.cache_resource
def get_collection():
    db = chromadb.PersistentClient(path=DB_PATH)
    embedder = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large",
    )
    return db.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
    )


collection = get_collection()
stored = collection.count()

st.caption(f"{len(chunk_files)} chunk files on disk — {stored} indexed in Chroma")

if stored != len(chunk_files):
    st.warning("The index does not match the chunk files.")

if st.button("Build or rebuild index"):
    with st.spinner("Indexing chunks..."):
        try:
            existing = collection.get()["ids"]
            if existing:
                collection.delete(ids=existing)

            collection.add(
                documents=[f.read_text(encoding="utf-8") for f in chunk_files],
                ids=[f.name for f in chunk_files],
            )
            st.success(f"Indexed {collection.count()} chunks")
            st.rerun()
        except Exception as error:
            st.error(f"Indexing failed: {error}")

if stored == 0:
    st.info("Build the index before asking questions.")
    st.stop()

st.divider()

question = st.text_input("Ask a question about the document:")

if st.button("Ask document"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching the document..."):
                results = collection.query(
                    query_texts=[question],
                    n_results=TOP_K,
                )

                documents = results["documents"][0]
                ids = results["ids"][0]
                distances = results["distances"][0]

                excerpts = "\n\n".join(
                    f"[{name}]\n{doc}" for name, doc in zip(ids, documents)
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
            for name, doc, dist in zip(ids, documents, distances):
                with st.expander(f"{name} — distance {dist:.3f}"):
                    st.write(doc)

        except Exception as error:
            st.error(f"Error: {error}")