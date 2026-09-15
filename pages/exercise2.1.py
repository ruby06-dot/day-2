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

MONTHS = (
    "January|February|March|April|May|June|"
    "July|August|September|October|November|December"
)

PRESETS = {
    "Square brackets — [1], [340]": r"(?m)^(?=\s*\[\d+\])",
    "Round brackets — (1), (2)": r"(?m)^(?=\s*\(\d+\))",
    "Bare numbers — 92 Accordingly": rf"(?m)^(?=\s*\d+\s+(?!(?:{MONTHS})\b)[A-Z])",
    "Dotted numbers — 1. / 1.1 / 1.1.1": r"(?m)^(?=\s*\d+(?:\.\d+)*\.\s)",
    "Any numbered style (mixed documents)": rf"(?m)^(?=\s*(?:\[\d+\]|\(\d+\)|\d+[\.\)]\s|\d+\s+(?!(?:{MONTHS})\b)[A-Z]))",
    "Lettered items — (a), (b), (c)": r"(?m)^(?=\s*\(?[a-z]\))",
    "Articles — Article 1, Article 2": r"(?m)^(?=\s*Article\s+\d+)",
    "Treaty style — Article / Rule / Regulation": r"(?m)^(?=\s*(?:Article|Rule|Regulation|Chapter|Section)\s+\d+)",
    "Sections — Section 1 / § 1": r"(?m)^(?=\s*(?:Section\s+\d+|§\s*\d+))",
    "Chapters or Parts": r"(?m)^(?=\s*(?:CHAPTER|Chapter|PART|Part)\s+[IVXLC\d]+)",
    "ALL CAPS headings": r"(?m)^(?=[A-Z][A-Z\s]{4,}$)",
}


def clean_text(raw):
    # Drop lines carrying several control characters (broken font encoding)
    raw = re.sub(r"(?m)^(?:[^\n]*[\x00-\x08\x0b-\x1f]){3,}[^\n]*$\n?", "", raw)
    # Strip any remaining stray control characters
    raw = re.sub(r"[\x00-\x08\x0b-\x1f]", "", raw)
    # Drop standalone page numbers
    raw = re.sub(r"(?m)^\s*\d{1,4}\s*$\n?", "", raw)
    # Rejoin words broken across lines
    raw = re.sub(r"-\n(\w)", r"\1", raw)
    raw = re.sub(r"\n\s*\n", "<<PARA>>", raw)
    raw = re.sub(r"\n(?=\s*(?:\[\d+\]|\(\d+\)|\d+\s+[A-Z]))", "<<PARA>>", raw)
    raw = raw.replace("\n", " ")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\s+\d{1,4}(?=\n|$)", "", raw)
    return raw.replace("<<PARA>>", "\n\n").strip()


def merge_out_of_sequence(chunks):
    """Merge chunks whose leading number breaks the ascending sequence."""
    merged = []
    last = 0
    for c in chunks:
        match = re.match(r"^\s*\[?(\d+)\]?\s", c)
        number = int(match.group(1)) if match else None
        if not merged:
            merged.append(c)
            if number is not None:
                last = number
        elif number is not None and number > last:
            merged.append(c)
            last = number
        else:
            merged[-1] = merged[-1].rstrip() + " " + c.lstrip()
    return merged


def chunk_label(body, fallback):
    match = re.match(r"^(\[\d+\]|\d+)(?=\s)", body)
    if match:
        tag = match.group(1)
        preview = body[len(tag):].strip()[:60]
    else:
        tag = f"#{fallback}"
        preview = body[:60]
    return f"{tag}  {preview}…  ({len(body)} chars)"


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

    for n, path in enumerate(saved, start=1):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        with st.expander(chunk_label(content, n)):
            st.markdown(content)
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
        counts = {k: len(re.findall(v, text)) for k, v in PRESETS.items()}
        labels = [f"{k}  ({counts[k]} matches)" for k in PRESETS]
        best = max(counts, key=counts.get)
        choice = st.selectbox(
            "Numbering style",
            labels,
            index=list(PRESETS).index(best),
        )
        key = choice.rsplit("  (", 1)[0]
        chunks = [c for c in re.split(PRESETS[key], text) if c.strip()]

        if st.checkbox("Merge numbers that break the sequence", value=True):
            before = len(chunks)
            chunks = merge_out_of_sequence(chunks)
            if before != len(chunks):
                st.caption(f"Merged {before - len(chunks)} false splits")

        if len(chunks) <= 1:
            st.warning("No match found. Try another style.")

    cleaned_chunks = [clean_text(c) for c in chunks]
    cleaned_chunks = [c for c in cleaned_chunks if c.strip()]

    st.write(f"{len(cleaned_chunks)} chunks")

    if cleaned_chunks:
        sizes = sorted(len(c) for c in cleaned_chunks)
        st.caption(
            f"Largest {sizes[-1]} chars · median {sizes[len(sizes) // 2]} chars"
        )
        if sizes[-1] > 30000:
            st.warning(
                "One chunk exceeds the embedding limit. Try a different style."
            )

    if st.button("Save chunks to disk"):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

        for n, c in enumerate(cleaned_chunks, start=1):
            path = os.path.join(OUTPUT_DIR, f"chunk_{n:03d}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(c)

        st.success(f"Saved {len(cleaned_chunks)} chunks")
        st.rerun()

    for n, c in enumerate(cleaned_chunks, start=1):
        with st.expander(chunk_label(c, n)):
            st.markdown(c)