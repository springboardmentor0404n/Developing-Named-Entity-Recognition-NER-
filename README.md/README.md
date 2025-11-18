FinanceInsight
Developing Named Entity Recognition (NER) Models for Financial Data Extraction

Virtual Internship 6.0 – Batch 3, Infosys Springboard


Project Overview

FinanceInsight is an end-to-end financial document understanding system designed to extract structured financial insights from unstructured documents such as annual reports, 10-K filings, and financial statements.

The system automates:

    1. Named Entity Recognition (NER) for financial entities
    2. Extraction of revenue, profit, expenses, ratios, companies, etc.
    3. Detection of financial events (M&A, IPO, stock split…)
    4. Document segmentation (MD&A, Risk Factors, Statements…)
    5. Parsing of financial tables from PDF reports
    6. Converting extracted information into a clean, structured format
    7. Providing an interactive Streamlit UI for analysis

This project was completed under Milestones 1 to 4 of the internship.

Repository Structure

FinanceInsight/
│
├── financial_extractor_app.py       # Streamlit UI
├── requirements.txt                 # Dependencies
├── README.md                        # You are here
│
├── Models/
│   ├── bert-ner-final/              # Final fine-tuned BERT model
│   └── bert-ner-transformer/        # Transformer checkpoints
│
├── Training/
│   ├── train.json
│   ├── valid.json
│   └── test.json
│
├── Data/
│   ├── sample_pdfs/                 # Example test PDFs
│   └── aapl_10k.html                # Extracted SEC example
│
├── Report/
│   └── FinanceInsight_Final_Report.pdf
│
└── Presentation/
    └── FinanceInsight_PPT.pdf

Results Summary : 

1. Successfully extracted meaningful financial text from 10-K filings
2. Segmented long annual reports (100+ pages)
3. Parsed multiple financial tables (tested on Nestlé Annual Report 2022)
4. Achieved high accuracy in NER for common financial entities
5. Built a fully interactive UI demonstrating the system end-to-end


Future Enhancements : 

1. Deploy online using Streamlit Cloud / Hugging Face Spaces
2. Add LLM-based summarization for MD&A
3. Add graph-based visualization (revenue, profit trends)
4. Support OCR extraction from scanned PDFs
5. Build cache for faster re-processing




Swati Upadhyay – Batch 3
Infosys Springboard Virtual Internship 6.0