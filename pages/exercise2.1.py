import streamlit as st
import re
import os
import glob
import shutil
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Exercise 2-1")

OUTPUT_DIR = "chunks"

PRESETS = {
    "Numbered paragraphs (1, 2, 3...)": r"(?m)^(?=\d+\s+[A-Z])",
    "Numbered clauses (1. / 1.1 / 1.1.1)": r"(?m)^(?=\s*\d+(?:\.\d+)*\.?\s)",
    "Articles (Article 1, Article 2...)": r"(?m)^(?=\s*Article\s+\d+)",
    "Sections (Section 1 / § 1)": r"(?m)^(?=\s*(?:Section\s+\d+|§\s*\d+))",
    "Chapters or Parts": r"(?m)^(?=\s*(?:CHAPTER|Chapter|PART|Part)\s+[IVXLC\d]+)",
    "Lettered items ((a), (b), (c)...)": r"(?m)^(?=\s*\(?[a-z]\))",
    "ALL CAPS headings": r"(?m)^(?=[A-Z][A-Z\s]{4,}$)",
}


def clean_text(raw):
    raw = re.sub(r"-\n(\w)", r"\1", raw)
    raw = re.sub(r"\n\s*\n", "<<PARA>>", raw)
    raw = re.sub(r"\n(?=\d+\s+[A-Z])", "<<PARA>>", raw)
    raw = raw.replace("\n", " ")
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.replace("<<PARA>>", "\n\n")


@st.cache_data(show_spinner=False)
def summarise(content):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=600,
        messages=[
            {
                "role": "system",
                "content": "You summarise documents accurately. Do not add facts that are not in the text.",
            },
            {
                "role": "user",
                "content": f"Summarise the key points of this document in bullet form.\n\n{content}",
            },
        ],
    )
    return response.choices[0].message.content


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

st.divider()

st.subheader("Saved chunks")
saved = sorted(glob.glob(f"{OUTPUT_DIR}/chunk_*.txt"))

if saved:
    st.write(f"{len(saved)} chunks on disk")

    if st.button("Clear saved chunks"):
        shutil.rmtree(OUTPUT_DIR)
        st.rerun()

    for path in saved:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        with st.expander(f"{os.path.basename(path)} — {len(content)} chars"):
            st.write(content)
else:
    st.caption("No saved chunks yet.")

st.divider()

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    st.write(f"Pages: {len(reader.pages)}")

    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    st.subheader("Summary")
    if st.button("Generate summary"):
        with st.spinner("Summarising..."):
            try:
                st.markdown(summarise(clean_text(text)[:40000]))
            except Exception as e:
                st.error(f"Summary failed: {e}")

    st.divider()
    st.subheader("Chunks")

    mode = st.radio(
        "Split method",
        ["By page", "By paragraph", "By document structure"],
    )

    if mode == "By page":
        chunks = [p.extract_text() or "" for p in reader.pages]

    elif mode == "By paragraph":
        cleaned = clean_text(text)
        chunks = [c for c in re.split(r"\n\s*\n", cleaned) if c.strip()]

    else:
        choice = st.selectbox("Document structure", list(PRESETS.keys()))
        chunks = [c for c in re.split(PRESETS[choice], text) if c.strip()]
        if len(chunks) <= 1:
            st.warning("No match found. Try another structure.")

    st.write(f"{len(chunks)} chunks")

    if st.button("Save chunks to disk"):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

        for n, c in enumerate(chunks, start=1):
            path = os.path.join(OUTPUT_DIR, f"chunk_{n:03d}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(clean_text(c).strip())

        st.success(f"Saved {len(chunks)} chunks")
        st.rerun()

    for i, c in enumerate(chunks):
        preview = clean_text(c).strip()[:40]
        with st.expander(f"Chunk {i+1} ({len(c)} chars) — {preview}"):
            st.markdown(clean_text(c))