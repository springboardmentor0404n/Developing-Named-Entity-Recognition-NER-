# 🚀 FinanceInsight: Automated Financial Document Understanding System  
### Developing Named Entity Recognition (NER) Models for Financial Data Extraction  
**Infosys Springboard Virtual Internship 6.0 — Batch 3**

---

## 🔍 Overview

**FinanceInsight** is an end-to-end financial document analysis system that processes large and unstructured financial PDFs such as Annual Reports, CSR Reports, and SEC 10-K filings.

The system extracts financial entities, tables, document sections, and insights using:

- 🧠 A custom **BERT-based NER model**
- 📄 PDF parsing (**pdfplumber**, **PyPDF2**)
- 📊 Table extraction engine
- 🖥️ Streamlit user interface

Created under the guidance of **Mr. G. Navinash**.

---

## 🎯 Objectives

- Extract financial entities: **Revenue, Profit, Ratios, Company Names**
- Identify financial events: **Dividends, Stock splits, Litigation, M&A**
- Parse complex financial tables: **Balance Sheet, P&L, Cash Flow**
- Segment documents into **MD&A**, **Risks**, **Highlights**, **Statements**
- Build a user-friendly **Streamlit app**
- Generate structured **JSON output**

---

## 🧠 Problem Statement

Financial PDFs are long, unstructured, and inconsistent across companies.

### Challenges include:
- Extracting entities from unstructured text  
- Detecting events inside narrative paragraphs  
- Parsing **multi-column**, **multi-page** tabular data  
- Identifying major financial sections  
- Building an intuitive UI  

---

## 🛠️ Solution Approach

FinanceInsight solves these challenges using a modular architecture:

### **1️⃣ PDF Extraction Layer**
- Cleans text  
- Flattens columns  
- Normalizes spacing  

### **2️⃣ Document Segmentation Layer**
Detects:
- MD&A  
- Risk Factors  
- Notes  
- Highlights  

### **3️⃣ Financial NER Layer (BERT-based)**
Extracts:
- Revenue  
- Expenses  
- Profit  
- Company names  
- Ratios  
- Dates  

### **4️⃣ Table Processing Layer**
- Uses **pdfplumber** + heuristics to extract multi-column & multi-page tables  

### **5️⃣ Streamlit UI Layer**
- Interactive PDF upload & analysis  

### **6️⃣ Output Layer**
- Generates structured **JSON + insights**

---

## 🧰 Technologies Used

### **Languages**
- Python 3.12+

### **NLP / ML**
- HuggingFace Transformers  
- PyTorch  
- Tokenizers  

### **PDF Processing**
- PyPDF2  
- pdfplumber  

### **Data Handling**
- pandas  
- numpy  

### **Frontend**
- Streamlit  

### **Version Control**
- Git & GitHub  

---

## 🖥️ Running the Application

### **1️⃣ Clone Repository**
```bash
git clone https://github.com/springboardmentor0404n/Developing-Named-Entity-Recognition-NER-.git
```

### **2️⃣ Navigate**
```bash
cd finance_ner_app
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
streamlit run financial_extractor_app.py
```

App opens at:  
👉 **http://localhost:8501**

---

## 📊 Results

### **Tested On**
- Tesla SEC 10-K (2023)  
- Nestlé Annual Report 2022  
- Sample Financial PDFs  

### **Extracted**
- ✔ Clean document text  
- ✔ Segmented sections  
- ✔ 43+ tables (Nestlé 2022)  
- ✔ Entities: Revenue, Profit, Company Names, Assets  
- ✔ JSON output  

---

## ✨ Key Features
- Multi-column PDF support  
- BERT-based Financial NER  
- Table extraction engine  
- Automatic section segmentation  
- Finance-specific insights  
- Streamlit drag-and-drop interface  
- JSON export  

---

## 🚀 Future Enhancements
- GPT-based summarization  
- Financial ratio calculator  
- Event classification  
- Sentiment Analysis (MD&A)  
- Year-over-year comparison  
- Multi-company benchmarking  

---

## 👩‍💻 Author
**Swati Upadhyay**  
Infosys Springboard — Batch 3  
Project: **FinanceInsight – Financial NER System**

---

## 🏁 Conclusion

FinanceInsight automates extraction of financial information from complex PDFs using:

- Deep learning  
- NLP  
- PDF parsing  
- Table extraction  
- Streamlit UI  

A complete **production-grade, end-to-end financial analytics pipeline**.

