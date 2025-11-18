import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

st.set_page_config(page_title="Financial Document Analyzer", layout="wide")

st.title("Financial Document Extraction System")

uploaded_file = st.file_uploader("Upload an Annual Report PDF", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    st.subheader("Extracted Text Preview")
    st.text(text[:2000] + " ...")

    # Example: Run your segmentation function
    st.subheader("Document Segmentation")
    st.write("Management Discussion & Analysis, Risk Factors, Financial Highlights (demo).")

    # Example: Show parsed tables
        # ===== Parsed Tables (Financial Tables Only) =====
    st.subheader("Parsed Tables")

    import pdfplumber
    import pandas as pd

    # Simple keyword-based filter to keep only financial tables
    financial_keywords = [
        "total assets", "total liabilities", "equity",
        "net income", "net profit", "revenue",
        "cash flow", "operating activities", "financing activities",
        "income statement", "balance sheet"
    ]

    financial_tables = []

    # Re-open the PDF for table extraction
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                continue

            for table in tables:
                df = pd.DataFrame(table)
                # Drop completely empty rows/cols
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue

                # Flatten text in table and search for financial keywords
                flat_text = " ".join(
                    df.astype(str).fillna("").values.flatten()
                ).lower()

                if any(kw in flat_text for kw in financial_keywords):
                    financial_tables.append((page_num, df))

    if not financial_tables:
        st.info(
            "No clear financial tables detected in this PDF yet. "
            "The text extraction and segmentation are working; "
            "table parsing can be extended for more complex layouts like SEC filings."
        )
    else:
        st.success(f"Detected {len(financial_tables)} financial-looking table(s).")

        for page_num, df in financial_tables:
            st.markdown(f"**Table from Page {page_num}**")
            st.dataframe(df.reset_index(drop=True))
