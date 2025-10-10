# Finance Insight  
**Financial Text Preprocessing & NLP Pipeline**

---

## 📘 Overview  
Finance Insight is an end-to-end NLP project that processes financial text data from multiple sources (news, SEC filings, reports, Wikipedia) and prepares it for **Named Entity Recognition (NER)** and language model training.

---

## ⚙️ Workflow  

1. **Data Unification** → Merge CSV, JSON, TXT, PDF, and HTML files  
   - ✅ 23,474 unique records  
   - 🗑️ 1.7M duplicates removed  

2. **Preprocessing** → Clean and normalize text  
   - Handles HTML tags, currencies, abbreviations, and date formats  

3. **Tokenization & POS Tagging** → Using spaCy (`en_core_web_sm`)  
   - Generates `linguistic_features.jsonl` and `token_stats.csv`  

4. **EDA & Visualization**  
   - Conducted via Jupyter notebooks:
     - `01_preprocessing_eda.ipynb`  
     - `02_eda_visualizations.ipynb`

---

## 🧰 Tech Stack  
`Python`, `spaCy`, `pandas`, `pdfplumber`, `BeautifulSoup`, `matplotlib`, `seaborn`, `tqdm`

---

## 📂 Structure  
Finance-Insight/
│
├── data/
│ ├── raw/ # Raw datasets (CSV, JSON, PDF, TXT)
│ ├── processed/ # Processed outputs and intermediate files
│ │ ├── merged_dataset.jsonl
│ │ ├── preprocessed_dataset.jsonl
│ │ ├── linguistic_features.jsonl
│ │ └── token_stats.csv
│ └──DATA_SOURCES.md
├── notebooks/
│ ├── 01_preprocessing_eda.ipynb # Compare before vs after cleaning
│ └── 02_eda_visualizations.ipynb # Charts & insights on tokens, length, etc.
│
├── scripts/
│ ├── prepare_dataset.py # Data unification from all formats
│ ├── preprocess_data.py # Text cleaning & domain normalization
│ ├── tokenize_features.py # Tokenization + POS + Lemmatization
│ └── legacy/ # Old versions for reference
│
├── requirements.txt
├── .gitignore
└── README.md

“From raw financial reports to clean insights — powered by NLP.” 