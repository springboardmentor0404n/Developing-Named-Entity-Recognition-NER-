# **FinanceInsight : AI-Powered Financial Document Intelligence System**
### Developing Named Entity Recognition (NER) Models for Financial Data Extraction  

---


![SCREENSHOT](<Screenshot 2025-11-20 181205.png>)


This project demonstrates real-world NLP + ML engineering, enabling end-to-end financial document intelligence.

---

## 🚀 **Project Overview**

Financial documents are long, unstructured, and complex.  
FinanceInsight solves this by automatically:

- Extracting key financial metrics (market cap, EPS, revenue growth, etc.)
- Detecting events like IPO, M&A, earnings announcements
- Parsing tables from PDF filings
- Document segmentation (MD&A, Risk Factors, Financial Statements, Notes)
- Yahoo Finance ticker verification
- Providing a Streamlit-powered UI to **chat with your financial documents**

---

## ⭐ **Key Features**

### 🔍 **1. Financial Named Entity Recognition (NER)**
Custom-trained FinBERT-based NER that extracts:

- Market Capitalization  
- EPS  
- Revenue Growth  
- PE Ratio  
- Price Trend  
- Other numeric financial indicators  

### 🧠 **2. Rule-Based Extraction**
Flexible regex + linguistic patterns handle:

- “EPS of $3.12”
- “Revenue grew 14% YoY”
- Multi-currency detection (USD, INR, EUR, GBP)

### 📈 **3. Financial Event Detection**
Identifies:

- **IPO**
- **M&A**
- **Earnings Call**
- **Dividend**
- **Rating Change**
- **Guidance / Forecasts**

### 📄 **4. PDF Parsing**
- Text extraction using `pdfplumber`
- Table extraction and reconstruction  
- Shift-correction for broken PDF cells  
- Table type classification:
  - Balance Sheet  
  - Income Statement  
  - Cash Flow  
  - Other  

### 🧩 **5. Document Segmentation**
Auto-detect sections like:

- Executive Summary  
- MD&A  
- Risk Factors  
- Financial Statements  
- Notes  

### 📉 **6. Yahoo Finance Verification**
For each detected ticker:

- Live stock price  
- Market cap  
- Sector & industry  
- 1M, 3M, 1Y returns  
- Comparison vs S&P500  

### 💬 **7. Chat With Your Document**
Upload a PDF → Ask questions → Get insights.

### 🤖 **8. LLM Integration (Planned)**
Future extension:

- Gemini-based Q&A  
- Section summaries  
- Embedding search  

---

## 🧰 **Tech Stack**

| Component | Technology |
|----------|------------|
| NER Model | Hugging Face + FinBERT |
| Backend | Python |
| PDF Parsing | PyPDF2, pdfplumber, pandas |
| Web App | Streamlit |
| Finance API | yfinance |
| ML | PyTorch |
| LLM (Planned) | Gemini API |
| Deployment | Streamlit Cloud |

---

## 📊 **Model Training Summary**

### **Dataset Size**
- Train: **9284**
- Validation: **1161**
- Test: **1161**

### **Training Performance**

| Epoch | Train Loss | Val Loss | Precision | Recall | F1 | Accuracy |
|------|------------|----------|-----------|--------|----|----------|
| 1 | 0.2997 | 0.3997 | 0.4572 | 0.7670 | 0.5729 | 0.8469 |
| 2 | 0.3082 | 0.4104 | 0.6152 | 0.8135 | 0.7006 | 0.9050 |
| 3 | 0.2004 | 0.5172 | 0.6513 | 0.8077 | 0.7211 | 0.9127 |

✔ High accuracy  
✔ Strong recall on numeric financial entities  

---

## 📂 **Project Structure**
```
Finance-Insight/
│
├── data/
│   ├── processed/
│   │   ├── ner_auto_splits/
│   │   ├── bio_annotation_ready.jsonl
│   │   ├── bio_auto_annotated.jsonl
│   │   ├── linguistic_features.jsonl
│   │   ├── merged_dataset.jsonl
│   │   ├── ner_auto_splits.zip
│   │   ├── preprocessed_dataset.jsonl
│   │   └── token_stats.csv
│   │
│   └── raw/
│       ├── filings/
│       ├── news/
│       ├── reports/
│       └── .gitkeep
│
├── models/
│   └── finbert_ner_weighted/
│       └── checkpoint-9284/
│
├── notebooks/
│   ├── 01_preprocessing_eda.ipynb
│   ├── 02_eda_visualizations.ipynb
│   ├── 03_finance_insight_model.ipynb
│   └── 04_segmentation_parsing.ipynb
│
├── scripts/
│   ├── legacy/
│   ├── 1_prepare_dataset.py
│   ├── 2_preprocess_pipeline.py
│   ├── 3_tokenize_features.py
│   ├── 4_prepare_bio_dataset.py
│   ├── 5_auto_annotate_and_sample.py
│   ├── 6_make_auto_hf_splits.py
│   └── 7_train_quick_ner.py
│
├── app.py
├── finance_insight_backend.py
├── README.md
├── requirements.txt
└── .gitignore
```


---

## ▶️ **How to Run Locally**

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/Suryasnata1404/Finance-Insight.git
```

### **2️⃣ Navigate**
```bash
cd Finance-Insight
```

### **3️⃣ Create Virtual Environment**
```bash
python -m venv venv
```

### **4️⃣ Activate**
```bash
venv\Scripts\activate
```

### **5️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **6️⃣ Run Streamlit App**
```bash
streamlit run app.py
```

App opens at:  
👉 **http://localhost:8501**

---

## 📊 Results

### **Tested On**
- Tesla SEC **10-Q Report (2023)**  
- Financial blog snippets  
- Market news paragraphs  
- Sample earnings summaries  

### **Extracted Successfully**
- ✔ Clean, fully-parsed document text  
- ✔ Accurate **section segmentation** (Executive Summary, MD&A, Risk Factors, Financial Statements)  
- ✔ **30+ structured tables** extracted & normalized (Tesla 10-Q)  
- ✔ **Entities Detected:** market_cap, EPS, revenue_growth, pe_ratio  
- ✔ **Event detection:** IPO, M&A, earnings_call  
- ✔ **Ticker verification (Yahoo Finance)** with price, market cap, sector, returns  
- ✔ Downloadable **JSON output** for all results  

### **Performance**
- Model trained on 11,600+ annotated samples  
- Achieved on Test Set:  
  - **Precision:** 0.65  
  - **Recall:** 0.81  
  - **F1 Score:** 0.72  
  - **Accuracy:** 0.91  

---

## 🚀 Future Enhancements
- **Gemini-powered document Q&A** (interactive financial assistant)  
- **Automated company financial scoring** (profitability, leverage, efficiency indexes)  
- **Multi-company comparison engine** (benchmarking & visualizations)  
- **Vision-based table extraction** (OCR + deep learning for scanned PDFs)  
- **Advanced segmentation using transformer models**  
- **Cross-document linking** (compare across years or filings)  
- **Smart anomaly detection** in financial statements  

---

## 👤 Author
**Suryasnata Mohapatra**  
Infosys Springboard — Batch 3  
Project: **FinanceInsight – Financial NER System**

GitHub: [https://github.com/Suryasnata1404](https://github.com/Suryasnata1404)  

---

## 📜 License
This project is licensed under the **MIT License**.


“From raw financial reports to clean insights — powered by NLP.” 