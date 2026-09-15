import streamlit as st
import re
import os
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Exercise 2-1")

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

    for i, c in enumerate(chunks):
        preview = clean_text(c).strip()[:40]
        with st.expander(f"Chunk {i+1} ({len(c)} chars) — {preview}"):
            st.markdown(clean_text(c))