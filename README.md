🚀 FinanceInsight: Automated Financial Document Understanding System
Developing Named Entity Recognition (NER) Models for Financial Data Extraction

Infosys Springboard Virtual Internship 6.0 — Batch 3

🔍 Overview

FinanceInsight is an end-to-end financial document analysis system that processes large and unstructured financial PDFs such as Annual Reports, CSR Reports, and SEC 10-K filings.

The system extracts financial entities, tables, document sections, and insights using:

A custom BERT-based NER model

PDF parsing (pdfplumber, PyPDF2)

Table extraction engine

Streamlit UI

Created under the guidance of Mr. G. Navinash.

🎯 Objectives

Extract financial entities: Revenue, Profit, Ratios, Company Names

Identify financial events: Dividends, Stock splits, Litigation, M&A

Parse complex financial tables (Balance Sheet, P&L, Cash Flow)

Segment documents into MD&A, Risks, Highlights, Statements

Build a user-friendly Streamlit app

Output structured JSON for downstream analytics

🧠 Problem Statement

Financial PDFs are long, unstructured, and inconsistent across companies.
Challenges include:

Extracting entities from free-text.

Detecting financial events buried inside narrative text.

Parsing multi-column, multi-page PDF tables.

Identifying major financial sections automatically.

Providing a clean UI for non-technical users.

🛠️ Problem Solution

FinanceInsight addresses the above challenges using a modular extraction framework:

PDF Extraction Layer
Cleans text, flattens columns, normalizes spacing.

Document Segmentation Layer
Detects MD&A, Risk Factors, Financial Highlights, Notes, etc.

Financial NER Layer (BERT-based)
Extracts revenue, expenses, profit, company names, ratios, dates.

Table Processing Layer
Extracts complex tables using pdfplumber + heuristics.

Frontend UI Layer
Built using Streamlit for interactive PDF upload & analysis.

Output Layer
Generates structured tables + JSON + insights.

🧱 System Architecture
         ┌──────────────┐
         │   PDF Upload  │
         └───────┬──────┘
                 │
        ┌────────▼─────────┐
        │  PDF Text Parser  │
        └────────┬─────────┘
                 │
     ┌───────────▼───────────┐
     │ Document Segmentation │
     └───────────┬───────────┘
                 │
     ┌───────────▼───────────┐
     │   Financial NER Model  │
     └───────────┬───────────┘
                 │
     ┌───────────▼───────────┐
     │  Table Extraction Engine │
     └───────────┬───────────┘
                 │
     ┌───────────▼───────────┐
     │ Streamlit UI + Output │
     └────────────────────────┘

🧰 Technologies Used
Languages

Python 3.12+

Libraries
NLP / ML

HuggingFace Transformers

PyTorch

Tokenizers

PDF Processing

PyPDF2

pdfplumber

Data

pandas

numpy

Frontend

Streamlit

Version Control

Git & GitHub

📁 Project Structure
FinanceInsight/
│── financial_extractor_app.py
│── requirements.txt
│── train.json
│── test.json
│── valid.json
│── label.json
│── README.md
└── (model folders removed due to GitHub size limit)

🖥️ Running the Application
1️⃣ Clone Repository
git clone https://github.com/springboardmentor0404n/Developing-Named-Entity-Recognition-NER-.git

2️⃣ Navigate
cd finance_ner_app

3️⃣ Create Virtual Environment
python -m venv venv

4️⃣ Activate
venv\Scripts\activate

5️⃣ Install Dependencies
pip install -r requirements.txt

6️⃣ Run Streamlit App
streamlit run financial_extractor_app.py


The app opens at:
👉 http://localhost:8501

📊 Results

Tested Successfully On:

Tesla SEC 10-K (2023)

Nestlé Annual Report 2022

Sample Financial PDFs

Extracted Results:

✔ Clean Document Text
✔ Segmented Sections
✔ 43+ Financial Tables (in Nestlé Report)
✔ Identified Entities: Revenue, Profit, Company Names, Assets
✔ Generated JSON Output

✨ Key Features

Full PDF analysis with multi-column support

BERT-based Financial NER

Table extraction engine

Automatic section segmentation

Finance-specific insights

Streamlit drag-and-drop interface

JSON export

🚀 Future Enhancements

GPT-Based financial summarization

Financial ratio calculator

Event classification

Sentiment Analysis (MD&A)

Year-over-year comparison

Multi-company benchmarking

👩‍💻 Author

Swati Upadhyay
Infosys Springboard — Batch 3
Project: FinanceInsight – Financial NER System

🏁 Conclusion

FinanceInsight successfully automates extraction of financial information from complex PDFs.
It integrates deep learning, PDF parsing, NLP, table extraction, and UI development into a strong, production-like system.
