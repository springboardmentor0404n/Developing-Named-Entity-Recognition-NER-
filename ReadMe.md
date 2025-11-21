📘 FinanceInsight

Python • NLP • Transformers • RAG • FAISS • Streamlit

A complete end-to-end AI system for automated extraction, analysis, and understanding of financial documents (Annual Reports, 10-K filings, Earnings Statements) using OCR, NER, Vector Search, and Retrieval-Augmented Generation (RAG) powered by Groq LLM.

🌟 Features
🔍 Document Intelligence

PDF Parsing (PyMuPDF for digital text)

OCR Extraction for scanned pages (Tesseract)

Table Extraction (pdfplumber)

🤖 Financial NER

Custom DistilBERT-based NER model

Trained on 70k+ financial news dataset

Extracts:
✔ ORG
✔ MONEY
✔ PRODUCT
✔ DATE
✔ PERCENT
✔ EVENT
✔ Revenue, Profit, EPS, Financial statements

🧠 LLM + RAG Pipeline

FAISS vector database for Top-K retrieval

Sentence-Transformers for embeddings

Groq Llama-3 reasoning with retrieved evidence

Reduces hallucinations to nearly 0%

📊 Interactive Dashboard

Built using Streamlit

PDF upload

Entity extraction

Table viewer

Custom keyword search

RAG-powered Q&A

Export results

⚡ Performance

Custom NER Accuracy: 98.93%

F1-Score: 0.9894

Works on 100+ page reports smoothly

Handles images, tables, long text, noisy scans

📋 Table of Contents

Demo

Installation

Quick Start

Usage Guide

Project Structure

Model Performance

Dataset Description

Documentation

Contributing

License

🎬 Demo
Sample Outputs

Entity extraction from real Annual Reports

Table extraction

OCR results

RAG-based financial Q&A

Dashboard preview

Pipeline Overview
PDF / Scanned Report
→ OCR / Text Extraction
→ Segmentation & Chunking
→ Embedding Generation
→ FAISS Vector Search
→ NER Extraction
→ Groq LLM Reasoning (RAG)
→ Verified Output + Dashboard

🚀 Installation
Prerequisites

Python 3.10+

GPU recommended (Colab / CUDA)

pip

Step 1: Clone Repository
git clone https://github.com/springboardmentor0404n/Developing-Named-Entity-Recognition-NER-/edit/NER/Vamshi-Perada/FinanceInsight.git
cd FinSight-AI

Step 2: Create Virtual Environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Add Your API Key (Groq LLM)

Create .env:

GROQ_API_KEY=your_api_key_here

⚡ Quick Start
1. Run Streamlit Dashboard
streamlit run app.py


The app opens at:

👉 http://localhost:8501

2. Use the Pre-trained NER Model

Your ner_json.pkl is located at:

models/ner/ner_json.pkl

3. RAG Query Example
from src.rag.answer import answer_query

response = answer_query("What is Apple’s 2023 revenue?")
print(response)

📖 Usage Guide
Web Application
Upload Financial PDF

Annual Report

10-K Form

Quarterly Report

Earnings Statement

Features

View extracted text

Extract financial entities

Extract tables

Custom keyword search

Ask questions using RAG+LLM

Download structured data

Backend Functions
PDF Extraction
from src.pdf.extract import extract_pdf
text = extract_pdf("report.pdf")

NER Extraction
from src.ner.model import extract_entities
entities = extract_entities(text)

RAG Query
from src.rag.pipeline import rag_pipeline
rag_pipeline("Summarize financial performance")

📁 Project Structure
FinSight-AI/
├── app.py                                # Streamlit Dashboard
├── requirements.txt
├── models/
│   ├── ner_json.pkl                      # Trained NER model
│   └── embeddings/                       # Sentence embeddings
│
├── src/
│   ├── pdf/                              # PDF + OCR processing
│   │   ├── extract.py
│   │   ├── ocr.py
│   │   └── tables.py
│   │
│   ├── ner/                              # Financial NER model
│   │   ├── model.py
│   │   ├── utils.py
│   │   └── training_config.json
│   │
│   ├── rag/                              # Vector search + LLM
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retrieve.py
│   │   └── answer.py
│   │
│   ├── finance/                          # Yahoo Finance validation
│   ├── utils/                            # Helpers
│   └── dashboard/                        # Streamlit UI components
│
├── data/
│   ├── parquet/                          # 7-year financial dataset
│   └── sample_reports/
│
└── outputs/                              # Extracted entities & results

📊 Model Performance
NER Model Evaluation
Metric	Score
Accuracy	0.9893
Precision	0.9895
Recall	0.9893
F1-Score	0.9894
Loss	0.0301
Real-World Extraction Examples

✔ Apple Inc.
✔ iPhone 14
✔ Revenue of $383B
✔ Profit $99.8B
✔ Q2 2023
✔ 5%

RAG Performance

Relevance improved dramatically

Hallucination reduced to near-zero

Stable responses even on 100+ page documents

🧠 Dataset Description

70,974 rows

169 columns

7 years of financial news

Converted from JSON → Parquet

Includes sentiment, emotions, companies, industries, stock prices

Perfect for financial NER training

📚 Documentation

docs/TECHNICAL_DOCUMENTATION.md

docs/USER_GUIDE.md

docs/API_REFERENCE.md

docs/MODEL_DETAILS.md

🧪 Testing
Run NER evaluation:
python test_ner.py

Test RAG:
python test_rag.py

Test Dashboard:
python test_app.py

🤝 Contributing

We welcome contributions!

Fork → Create Branch → Commit → Push → Pull Request

📝 License

This project is licensed under the MIT License.

🙏 Acknowledgements

HuggingFace Transformers

Groq LLM API

Sentence Transformers

FAISS

PyMuPDF, pdfplumber

Tesseract OCR
