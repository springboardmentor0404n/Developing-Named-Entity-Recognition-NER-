import streamlit as st
import pdfplumber
import camelot
import re
import pandas as pd
import yfinance as yf


# --------------------------
# Extract TEXT
# --------------------------
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


# --------------------------
# Extract TABLES
# --------------------------
def extract_pdf_tables(pdf_path):
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        return tables
    except:
        return None


# --------------------------
# SIMPLE ENTITY EXTRACTION
# --------------------------
def extract_entities(text, user_entities):
    extracted = {}

    patterns = {
        "revenue": r"(revenue|income)\s*[:\-]?\s*\$?\s*([\d\.]+[MBK]?)",
        "eps": r"(EPS|earnings per share)\s*[:\-]?\s*\$?\s*([\d\.]+)",
        "market cap": r"(market cap|market capitalization)\s*[:\-]?\s*\$?\s*([\d\.]+[MBT]?)",
        "profit": r"(profit|net profit)\s*[:\-]?\s*\$?\s*([\d\.]+[MBK]?)",
        "growth": r"(growth|increase)\s*[:\-]?\s*([\d\.]+%)"
    }

    for ent in user_entities:
        ent = ent.lower().strip()
        if ent in patterns:
            match = re.findall(patterns[ent], text, re.IGNORECASE)
            extracted[ent] = [m[1] for m in match] if match else ["Not found"]

    return extracted



# --------------------------
# STREAMLIT APP UI
# --------------------------
st.title(" Financial Named Entity Recognition System")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

entities_list = ["revenue", "eps", "market cap", "profit", "growth"]

selected_entities = st.multiselect(
    "Select financial entities to extract:",
    entities_list
)

ticker = st.text_input("Enter Ticker (Optional):")


if st.button("Extract"):
    if not uploaded_file:
        st.error("Upload a PDF first.")
        st.stop()

    st.success("Processing...")

    # Save uploaded PDF for Camelot
    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # ------------------------
    # TEXT EXTRACTION
    # ------------------------
    text = extract_pdf_text(temp_pdf_path)
    st.subheader("📄 Text Preview")
    st.write(text[:700] + " ...")

    # ------------------------
    # ENTITY EXTRACTION
    # ------------------------
    if selected_entities:
        results = extract_entities(text, selected_entities)
        st.subheader("📌 Extracted Entities")
        st.write(results)
    else:
        st.warning("No entities selected.")

    # ------------------------
    # TABLE EXTRACTION
    # ------------------------
    st.subheader("📊 Extracted Tables")
    tables = extract_pdf_tables(temp_pdf_path)

    if tables and tables.n > 0:
        for i, table in enumerate(tables):
            st.write(f"Table {i+1}")
            df = table.df
            st.dataframe(df)
    else:
        st.info("No tables found or Camelot could not read this PDF.")

    # ------------------------
    # OPTIONAL MARKET DATA
    # ------------------------
    if ticker:
        st.subheader("📈 Market Data")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            st.write({
                "Market Cap": info.get("marketCap"),
                "EPS": info.get("trailingEps"),
                "Revenue": info.get("totalRevenue")
            })
        except:
            st.error("Couldn't fetch market data.")
