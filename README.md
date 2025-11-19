<p align="center">
  <img src="assets/logo.png" alt="Project Logo" height="160">
</p>

<h1 align="center">FinanceInsight – AI-Powered Financial Document Analysis</h1>

<p align="center">
  <b>End-to-end NLP + ML pipeline for parsing, segmenting and analyzing financial reports</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen">
  <img src="https://img.shields.io/badge/Python-3.10-blue">
  <img src="https://img.shields.io/badge/Streamlit-1.32-red">
  <img src="https://img.shields.io/badge/Docker-Ready-brightgreen">
  <img src="https://img.shields.io/badge/License-MIT-yellow">
</p>

---

## 📌 Overview

**FinanceInsight** is an end-to-end AI system to analyze financial documents like **Annual Reports, 10-K Filings, Investor Reports**, etc.  
It performs automated:

- **Text Extraction**
- **Report Segmentation**
- **Table Extraction**
- **Event Extraction (NER-based)**
- **Sentiment Analysis (FinBERT)**
- **Dashboard Visualization**
- **Docker Deployment**

The project includes a full ML pipeline + a Streamlit dashboard + production-level Docker setup.

---

## 🎥 Demo (GIF Preview)

> Replace `assets/demo.gif` with your own recording.

<p align="center">
  <img src="assets/demo.gif" alt="Demo GIF" width="750">
</p>

---

## 🚀 Features

### 🔎 NLP & ML Pipeline
- PDF segmentation by headings  
- Table extraction & classification  
- Clean text preprocessing  
- Event extraction with transformer model  
- Sentiment analysis using FinBERT  

### 📊 Financial Dashboard (Streamlit)
- View segmented sections  
- View extracted tables  
- Visualize events and insights  
- Fully interactive UI  

### 🐳 Docker Deployment
- One-command build  
- Works on any server  
- Ready for Render / AWS / DigitalOcean  

---

## 📁 Project Structure

```bash
FinanceInsight/
│
├── app/
│   ├── streamlit_app.py        # Dashboard UI
│   ├── Dockerfile              # Production container
│   └── requirements.txt
│
├── scripts/
│   ├── 01_preprocess_fiqa.py
│   ├── 02_eda_fiqa.py
│   ├── 03_event_extraction.py
│   ├── 04_augment_data.py
│   ├── 05_segment_reports.py
│   ├── 06_parse_tables.py
│   ├── 07_eval_pipeline.py
│   ├── financial_entity_event_extractor.py
│   └── test_model.py
│
├── outputs/
│   ├── doc_segments/
│   ├── tables/
│   └── events/
│
├── data/
│   └── processed/
│
├── sample_reports/
│   └── 10K_sample.pdf
│
├── README.md
└── LICENSE
🛠 Installation (Local)
1️⃣ Clone the repository
git clone https://github.com/yogender-kumar-creator/FinanceInsight.git
cd FinanceInsight

2️⃣ Install dependencies
pip install -r app/requirements.txt

3️⃣ Run Streamlit dashboard
streamlit run app/streamlit_app.py

🐳 Docker Deployment
1️⃣ Build Docker image
docker build -t financial-dashboard ./app

2️⃣ Run container
docker run -p 8501:8501 \
  -v A:/Infosys/outputs:/app/outputs \
  financial-dashboard


Then open:

👉 http://localhost:8501

📘 How It Works (Pipeline)
1️⃣ PDF Segmentation
python scripts/05_segment_reports.py

2️⃣ Extract Tables
python scripts/06_parse_tables.py

3️⃣ Extract Events
python scripts/03_event_extraction.py

4️⃣ Evaluate Pipeline
python scripts/07_eval_pipeline.py

5️⃣ View on Dashboard

Output automatically appears in Streamlit.

🧪 Example Results

Total Segments: ✓

Total Tables Extracted: ✓

Events Found: ✓

VERIFIED Events: ✓

Errors (NER): ✓

(Your evaluation summary is included inside the repo)

🤝 Contribution Guidelines

You are welcome to contribute 🎉

✔ Fork the repo
✔ Create a feature branch
git checkout -b feature-name

✔ Commit changes
git commit -m "Added new feature"

✔ Push branch
git push origin feature-name

✔ Submit Pull Request
📜 License

This project is released under the MIT License.
See LICENSE file for full text.

⭐ Show Support

If this repo helped you, please ⭐ the repository.
Your support motivates more open-source work ❤️
