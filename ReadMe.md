FinanceInsight — Financial Named Entity Recognition 

This project is part of the AI Internship Program  and focuses on building a Financial Named Entity Recognition (NER) model using the FiNER-139 dataset.DOWNLOAD_URL = "https://huggingface.co/datasets/nlpaueb/finer-139/resolve/main/finer139.zip"
The main goal is to identify financial entities such as:

Company names

Stock tickers

Index names

Market indicators

Currency values

Financial instruments

Ratios, revenue, profits

Domain-specific financial concepts

🚀 Project Overview

Traditional NER models struggle with financial text due to:

Domain-specific terminology

Long entity names

Numeric-heavy tokens

Mixed patterns like “EPS”, “12-month yield”, “Q4 revenue”, etc.

This project builds a custom transformer-based NER model fine-tuned on the FiNER-139 dataset to extract rich entity information from financial documents.
Tech Stack
Programming	:Python
NLP Framework	:HuggingFace Transformers
Dataset	:FiNER-139 (HuggingFace)
Model	:BERT-Base-Cased (Token Classification)
Visualization	:Matplotlib, Seaborn
Metrics:	SeqEval (precision/recall/F1)

The pipeline includes:

✔️ Week 1–2 — Data Downloading + EDA

Download FiNER-139 dataset

Extract ZIP

Load JSONL files

Basic sanity checks

Token distribution plots

Entity frequency analysis

Numeric-token analysis

Export CSV/JSON artifacts

 Week 3–4 — Dataset Processing + Tokenization + Model Training

Convert FiNER-139 to  Datasets format

Tokenize with BERT tokenizer

Align word-level labels to subword tokens

Train a BERT-based Token Classification model

Evaluate using seqeval

Save model for inference

 Week 5–6 — Inference + Demo App

Run predictions on new financial text

Convert model outputs to human-readable entities

Build a small UI (Streamlit OR CLI-based depending on your choice)

Integration-ready for frontend usage
Future Improvements

Train with RoBERTa or FinBERT

Add CRF layer for better sequence modeling

Include more financial datasets

Convert backend to FastAPI

Deploy on HuggingFace Spaces
