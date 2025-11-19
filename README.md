📘 Financial Insights Extraction System

Retrieval-Augmented Financial Entity Extraction Pipeline



This project implements an end-to-end pipeline for retrieving financial text from a dataset and extracting structured financial information such as EPS, market capitalization, revenue mentions, and tickers.

It uses Sentence Transformers, FAISS, custom regex-based NER, and yFinance for fallback financial data retrieval.



🚀 Features



1\. Semantic Retrieval (Dense Retrieval)



Uses all-MiniLM-L6-v2 SentenceTransformer model



Embeds dataset into vector space



FAISS Inner Product Index for fast similarity search



Supports rebuilding or reusing saved embeddings (embeddings.npy) and index (faiss\_index.idx)



2\. Financial Entity Extraction



Custom extraction module identifies:



EPS (Earnings Per Share)



EPS currency detection



EPS trends (↑ increased, ↓ decreased)



Market Cap (supports units: billion, million, K, etc.)



Revenue growth mentions



Ticker extraction (NYSE, NASDAQ-style)



Ticker validation using yFinance



3\. External Data Enhancements



yFinance fallback



trailingEps



forwardEps



shortName



Yahoo Finance Search API 





4\. End-to-End Pipeline



Given a query:



Retrieve relevant financial sentences



Extract financial entities



Infer canonical EPS



Store results as retrieve\_outputs/\*.csv



Display human-readable summary in terminal



📂 Project Structure



finance-project/

│

├── retrieve\_and\_extract.py     # Main pipeline script (single-file)

├── financeinsight\_labeled\_with\_positive.csv (Not included in repo)

├── embeddings.npy              # Auto-generated (DO NOT COMMIT)

├── faiss\_index.idx             # Auto-generated (DO NOT COMMIT)

├── chunks\_meta.json            # Auto-generated

├── retrieve\_outputs/           # Generated output CSVs

└── README.md





🛠 Installation

Create conda environment

conda create -n finbert\_env python=3.10

conda activate finbert\_env



Install dependencies

pip install sentence-transformers faiss-cpu pandas scikit-learn tqdm yfinance requests





(Optional for faster CPU performance)



pip install -U numpy





▶️ Usage



Basic Retrieval + Extraction

python retrieve\_and\_extract.py --query "earnings per share Apple"



Force specific ticker



Useful when query doesn’t contain a ticker:



python retrieve\_and\_extract.py --query "earnings per share" --ticker AAPL



Rebuild embeddings \& FAISS index

python retrieve\_and\_extract.py --query "market cap Tesla" --rebuild





📊 Outputs

Example terminal summary:

\[807] idx=807  score=0.7115

Earnings per share increased significantly this quarter.

&nbsp; eps: \[]

&nbsp; eps\_trend: \['increased']

&nbsp; market\_cap: \[]

&nbsp; revenue: \[]

&nbsp; tickers\_raw: \[]  tickers\_valid: \[]



