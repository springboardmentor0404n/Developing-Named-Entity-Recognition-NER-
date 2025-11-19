FINAL PPT CONTENT – FinanceInsight Project
Slide 1 – TITLE

FinanceInsight: Financial Named Entity Recognition System
Infosys Springboard Internship – 2 Months Project

Slide 2 – TABLE OF CONTENTS

TABLE OF CONTENTS

Project Overview

Dataset Overview and Key Insights

Methodology
• Exploratory Data Analysis
• Visualization
• Data Preprocessing
• Feature Extraction
• Model Architecture
• Training and Evaluation

Conclusion

Results

Slide 3 – PROJECT OVERVIEW

Developed a Financial Named Entity Recognition (NER) system

Extracts key financial information like:

Company names, stock price, P/E ratio

Financial metrics (EBITDA, EPS, revenue, market cap)

Financial events (IPO, mergers, acquisitions)

Built using Milestone-based approach

Used NLP + Machine Learning + FinBERT/BERT

Achieved 98.24% accuracy with 3 epochs & batch size = 5

Helps analysts extract structured financial data automatically from text

Slide 4 – DATASET OVERVIEW & KEY INSIGHTS (EDA)

Dataset Used: constituents-financials.csv
Key Columns:

Company Name, Sector

Stock Price

Price/Earnings

Dividend Yield

Earnings/Share

Market Cap

EBITDA

52 Week High / Low

Key Insights from EDA

Found patterns in financial entities

Cleaned dataset and handled missing values

Identified most common financial terms

Prepared dataset for NER-based classification training

Visualizations: bar charts, word clouds, distribution plots

Slide 5 – METHODOLOGY (Workflow Overview)
DATA PREPROCESSING

Tokenization, normalization, lemmatization

Removed stopwords & unnecessary symbols

Handled ₹ / $ / % / € / crore / million

Financial terminology cleanup (EPS, P/E, EBITDA)

FEATURE EXTRACTION

POS tagging (Parts of Speech)

Word embeddings

Domain-specific vocabulary

Entity labeling for model training

MODEL ARCHITECTURE EXTRACTION

Models tested:

CRF (baseline)

Bi-LSTM + CRF

Transformer-based BERT / FinBERT

Fine-tuned model for financial text

Used attention mechanism and contextual embeddings

TRAINING & EVALUATION

Epochs: 3

Batch Size: 5

Accuracy Achieved: 98.24%

Evaluation Metrics:

Precision

Recall

F1-Score

Used classification report & confusion matrix

Slide 6 – MODEL ARCHITECTURE

Architecture Pipeline:

Input Financial Text
        ↓
Preprocessing (tokenization, cleaning)
        ↓
Word Embeddings (BERT)
        ↓
NER Model (FinBERT / BiLSTM-CRF)
        ↓
Entity Prediction (Company, Stock, Revenue, EPS…)
        ↓
Custom Extraction Module (events, metrics)
        ↓
Final Structured Output (JSON / CSV)


Advantages:

Domain-specific model

Better context understanding

High accuracy for financial entity extraction

Slide 7 – TRAINING & ACCURACY
Metric	Value
Accuracy	98.24%
Epochs	3
Batch Size	5
Recall	(from code)
Precision	(from code)
F1 Score	(from code)

✔ Used BERT / FinBERT fine-tuning
✔ Used Adam Optimizer + Learning Rate Scheduler
✔ Stability achieved using Gradient Clipping

Slide 8 – CONCLUSION

Developed end-to-end Financial NER system

Successfully extracted structured financial insights

Custom extraction module for EPS, revenue, market cap, events

Can improve fintech automation, stock analysis, report generation

Model can be deployed into dashboard / webapp / API service

Future Scope:

Real-time stock news monitoring

Live financial event extraction

Integration with Bloomberg / Yahoo Finance API
