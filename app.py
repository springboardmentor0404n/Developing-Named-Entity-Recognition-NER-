import streamlit as st
import pdfplumber
import camelot
import pandas as pd
import os

st.set_page_config(page_title="FinanceInsight | Financial NER & Table Extraction", layout="wide")

st.title("📊 FinanceInsight – Financial Document Analyzer")
st.write("Upload a financial PDF to extract text and tables.")

uploaded_file = st.file_uploader("Upload Annual Report PDF", type=["pdf"])

if uploaded_file:
    pdf_path = "uploaded.pdf"
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    # TEXT EXTRACTION
    with st.spinner("Extracting text..."):
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    full_text += t + "\n"

    st.subheader("📘 Extracted Text")
    st.text_area("Full Text", full_text[:4000], height=300)
    
    # TABLE EXTRACTION
    st.subheader("📈 Extract Financial Tables")
    pages_to_read = st.text_input("Enter pages (e.g., 100-140)", "100-140")

    if st.button("Extract Tables"):
        with st.spinner("Extracting tables using Camelot..."):
            tables = camelot.read_pdf(pdf_path, pages=pages_to_read, flavor="stream")

        st.write(f"Total tables found: {tables.n}")

        output_dir = "extracted_tables"
        os.makedirs(output_dir, exist_ok=True)

        for i, table in enumerate(tables):
            csv_path = os.path.join(output_dir, f"table_{i+1}.csv")
            table.df.to_csv(csv_path, index=False)
            st.write(f"Saved Table {i+1} to {csv_path}")
            st.dataframe(table.df)

    st.success("Processing complete!")
