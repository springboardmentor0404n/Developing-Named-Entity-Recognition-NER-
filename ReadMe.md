

FinanceInsight
Financial Named Entity Recognition (NER) and PDF Extraction System

Overview
FinanceInsight is an end-to-end financial information extraction system designed to process financial documents such as annual reports, filings, earnings reports, and structured PDFs.
The system uses a custom-trained NER model to extract key financial entities such as company names, stock tickers, revenue, market cap, dates, financial metrics, and financial events.
It also supports PDF ingestion, text extraction, and a simple interface for testing the model.

This project is built for analysts, researchers, and students who want automated extraction of financial insights from large text and semi-structured documents.

Features

1. Financial Named Entity Recognition (NER)

   * Extracts financial entities such as revenue, net income, EPS, stock tickers, company names, dates, events, and numeric values
   * Trained using domain-specific datasets
   * Supports inference on any financial text

2. PDF Text Extraction

   * Reads PDF files using pdfplumber
   * Extracts continuous text from multi-page documents
   * Supports scanned PDFs with optional OCR fallback (pytesseract)

3. Simple NER Inference Pipeline

   * Upload or load any text/PDF
   * Runs your custom NER model
   * Outputs entity spans, labels, and confidence scores

4. Web Demo Interface (Gradio)

   * Upload a PDF
   * Automatically extract text
   * View NER predictions in real-time
   * Deployable on Hugging Face Spaces or Cloud platforms

5. Model Reusability

   * Model saved using save_pretrained()
   * Can be loaded anytime locally or deployed on cloud
   * Compatible with CPU and GPU environments



Model Training Summary
The NER model was trained in four milestones:

Milestone 1: Data Preparation

* Collected financial texts (reports, filings, news)
* Preprocessed using tokenization, cleaning, normalization
* Performed EDA, augmentation (back-translation, masking)

Milestone 2: Financial NER Model

* Models explored: CRF, BiLSTM-CRF, BERT, FinBERT
* Final model fine-tuned on financial datasets
* Evaluated using precision, recall, F1-score
* Error analysis performed for misclassified financial terms

Milestone 3: Custom Financial Extraction

* User-defined metrics extraction (EPS, revenue)
* Event detection (mergers, earnings calls)
* Integration with financial APIs for validation

Milestone 4: PDF Processing and Segmentation

* Extracted text and tables from PDFs
* Applied NER on extracted text
* Produced structured JSON output

Deployment
The project supports multiple deployment methods:

1. Gradio Live (Colab)
2. HuggingFace Spaces
3. FastAPI + Docker + Google Cloud Run
4. Local CPU/GPU inference

The Gradio version is the simplest to deploy for demo purposes.

How to Save and Reload the Model
Save:

```python
model.save_pretrained("./financial_ner_model")
tokenizer.save_pretrained("./financial_ner_model")
```

Load later:

```python
tokenizer = AutoTokenizer.from_pretrained("./financial_ner_model")
model = AutoModelForTokenClassification.from_pretrained("./financial_ner_model")
```

Future Improvements

* Add OCR for scanned PDFs
* Add table extraction and LLM-based parsing
* Train a better domain-specific financial dataset
* Add UI dropdowns for selecting entity types
* Improve deployment performance with ONNX optimization


