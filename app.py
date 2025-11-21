# ================================================================
# 📘 FINANCIAL DOCUMENT INTELLIGENCE SYSTEM (FINAL Streamlit APP)
# ================================================================

# Add early environment checks and friendlier error messages
import sys
import textwrap

# Basic interpreter check
if sys.version_info < (3, 9):
    print(textwrap.dedent("""
    ERROR: Python 3.9+ is required to run this application.
    - Install a supported Python version from: https://www.python.org/downloads/
    - During install, check "Add Python to PATH".
    After installing, open a NEW terminal and run:
      python --version
    Then install dependencies and run:
      python -m venv .venv
      .venv\\Scripts\\activate
      pip install --upgrade pip
      pip install streamlit sentence-transformers transformers faiss-cpu PyMuPDF pdfplumber pytesseract pillow yfinance groq tabulate
      streamlit run app.py
    """))
    sys.exit(1)

# Try to import required packages and show actionable messages on failure
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import fitz
    import pdfplumber
    import pytesseract
    from PIL import Image
    import re, io, pickle, os
    import faiss
    import yfinance as yf

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    from groq import Groq
except Exception as e:
    print("ERROR: Failed to import required Python packages.")
    print("Cause:", e)
    print(textwrap.dedent("""
    Suggested fix:
    1) Ensure Python is installed and on PATH (see https://www.python.org/downloads/).
    2) Create and activate a virtual environment:
         python -m venv .venv
         .venv\\Scripts\\activate
    3) Install dependencies:
         pip install --upgrade pip
         pip install streamlit sentence-transformers transformers faiss-cpu PyMuPDF pdfplumber pytesseract pillow yfinance groq tabulate
    4) Run the app via Streamlit:
         streamlit run app.py
    If you still see the Microsoft Store message when running 'python', disable the Python app execution alias:
      Settings > Apps > Advanced app settings > App execution aliases
    """))
    sys.exit(1)

# ================================================================
# CONFIG
# ================================================================
st.set_page_config(page_title="Financial Analyzer", layout="wide")
st.title("📘 Financial Document Intelligence System")

NER_PKL = "ner_json.pkl"     # <-- place your trained model here
EMB_MODEL_NAME = "all-MiniLM-L6-v2"

# ================================================================
# LOAD NER MODEL
# ================================================================
@st.cache_resource
def load_ner_pipeline():
    with open(NER_PKL, "rb") as f:
        loaded = pickle.load(f)

    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-base-NER",
            num_labels=len(loaded["id2label"]),
            id2label=loaded["id2label"],
            label2id=loaded["label2id"]
        )
        model.load_state_dict(loaded["model_state_dict"], strict=False)
        return pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
    else:
        return loaded  # already a pipeline

ner_pipeline = load_ner_pipeline()

# ================================================================
# LOAD EMBEDDING MODEL
# ================================================================
embedder = SentenceTransformer(EMB_MODEL_NAME)

# ================================================================
# GROQ API
# ================================================================
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def groq_answer(question, chunks):
    context = "\n\n".join(chunks[:5])
    prompt = f"""
Use ONLY this financial report context:

{context}

QUESTION: {question}
"""

    output = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a precise financial assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return output.choices[0].message.content

# ================================================================
# HELPERS
# ================================================================
def extract_pdf_pages(pdf_bytes):
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, p in enumerate(doc):
            text = p.get_text("text") or ""
            if len(text.strip()) < 30:
                pix = p.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            pages.append({"page": i+1, "text": text})
    return pages

def chunk_text(pages, max_chars=800, overlap=200):
    chunks = []
    for p in pages:
        clean = re.sub(r"\s+", " ", p["text"]).strip()
        start = 0
        while start < len(clean):
            end = min(len(clean), start + max_chars)
            chunks.append(clean[start:end])
            start = end - overlap
            if start < 0:
                start = 0
            if end == len(clean):
                break
    return chunks

def build_faiss(chunks):
    vectors = embedder.encode(chunks, convert_to_numpy=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors.astype("float32"))
    return index, vectors

def search(query, chunks, index):
    qv = embedder.encode([query], convert_to_numpy=True)
    scores, idx = index.search(qv.astype("float32"), 5)
    return [chunks[i] for i in idx[0]]

def yahoo_lookup(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "Name": info.get("shortName"),
            "Market Cap": info.get("marketCap"),
            "EPS": info.get("trailingEps"),
            "Currency": info.get("currency"),
        }
    except:
        return {"error": "Invalid or unavailable ticker"}

# ================================================================
# UI
# ================================================================
uploaded = st.file_uploader("Upload Annual Report PDF", type=["pdf"])
ticker = st.sidebar.text_input("Yahoo Finance Ticker (optional):")

if uploaded:
    raw = uploaded.read()
    st.subheader("📄 Extracting PDF...")
    pages = extract_pdf_pages(raw)

    preview = "\n\n".join(p["text"] for p in pages)[:1500]
    st.text_area("Extracted Text Preview", preview, height=200)

    # ============================================================
    # NER
    # ============================================================
    st.header("🧠 NER: Extracted Financial Entities")
    ents = []
    for p in pages:
        res = ner_pipeline(p["text"])
        for r in res:
            ents.append({
                "page": p["page"],
                "entity": r["entity_group"],
                "word": r["word"].replace("##",""),
                "score": float(r["score"])
            })
    st.dataframe(pd.DataFrame(ents))

    # ============================================================
    # TABLE EXTRACTION
    # ============================================================
    st.header("📊 Extracted Financial Tables")
    financial_keywords = [
        "revenue", "net income", "profit", "assets",
        "liabilities", "cash flow", "equity"
    ]
    tables_found = []

    with pdfplumber.open(uploaded) as pdf:
        for i, p in enumerate(pdf.pages):
            try:
                tables = p.extract_tables()
            except:
                continue
            for tb in tables:
                df = pd.DataFrame(tb).dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue
                flat = " ".join(df.astype(str).values.flatten()).lower()
                if any(k in flat for k in financial_keywords):
                    tables_found.append((i+1, df))

    if tables_found:
        for pg, df in tables_found:
            st.subheader(f"Table from Page {pg}")
            st.dataframe(df.reset_index(drop=True))
    else:
        st.info("No financial tables detected.")

    # ============================================================
    # RAG + GROQ QA
    # ============================================================
    st.header("💬 Ask a Question (RAG + Groq LLM)")
    chunks = chunk_text(pages)
    index, _ = build_faiss(chunks)

    question = st.text_input("Ask about this financial document:")
    if question:
        retrieved = search(question, chunks, index)
        answer = groq_answer(question, retrieved)
        st.success(answer)
        with st.expander("Retrieved Context"):
            for i, c in enumerate(retrieved, 1):
                st.markdown(f"**Chunk {i}:** {c}")

    # ============================================================
    # YAHOO FINANCE VERIFICATION
    # ============================================================
    st.header("📈 Yahoo Finance Verification")
    if ticker:
        st.json(yahoo_lookup(ticker))

else:
    st.info("Upload a PDF to start.")
