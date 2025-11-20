# Financial Entity Recognition and Report Generator

## 📌 Project Overview
This project automates the extraction of financial metrics and entities from complex financial documents (e.g., 10-K reports).  
It generates structured JSON reports and visual charts, enabling faster analysis and reducing manual effort.

## 🚀 Features
- Document segmentation (MD&A, Risk Factors, Financial Statements)
- Entity extraction using NLP (spaCy)
- Table parsing and numeric conversion
- Company enrichment (sector, HQ, market cap)
- JSON report generation
- Chart visualization

## 🏗️ Architecture
Input Document → Segmentation → Entity Extraction → Table Parsing → Company Enrichment → Report Generation → Outputs: Charts + JSON

## 📂 Folder Structure
📁 FinanceInsight_Dataset  
├── runner.py  
├── segmentation.py  
├── table_parser.py  
├── nlp_pipeline.py  
├── integration.py  
├── report_generator.py  
├── visualization.py  
├── reports/  
│   ├── charts/  
│   └── json/  

## ⚙️ How to Run
```bash
python runner.py



📊 Outputs
Charts: reports/charts/*.png

JSON: reports/json/report.json

💡 Business Impact
Saves analysts time by automating manual review

Provides structured insights for faster decision-making

Scales across industries for compliance and audit
✅ Updated by Likitha

