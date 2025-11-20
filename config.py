# config.py
SECTION_ANCHORS = [
    "Management's Discussion and Analysis",
    "Risk Factors",
    "Financial Statements",
    "Notes",
    "Management Report",
    "Budget",
    "Revenue"
]

CURRENCY_SYMBOLS = ["$", "€", "₹"]
NULL_TOKENS = ["N/A", "NA", "Nil", "—", "-", "None"]
UNITS = ["B", "M", "K", "million", "billion", "thousand"]

# Regex components
NUMBER_PATTERN = r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?"

ROW_PATTERNS = {
    "Revenue": rf"Revenue[:\s]+(?:[$€₹])?\s*({NUMBER_PATTERN})\s*(?:B|M|K|million|billion|thousand)?",
    "Net Income": rf"Net\s+Income[:\s]+(?:[$€₹])?\s*({NUMBER_PATTERN})\s*(?:B|M|K|million|billion|thousand)?",
    "Operating Income": rf"Operating\s+Income[:\s]+(?:[$€₹])?\s*({NUMBER_PATTERN})\s*(?:B|M|K|million|billion|thousand)?",
    "Gross Profit": rf"Gross\s+Profit[:\s]+(?:[$€₹])?\s*({NUMBER_PATTERN})\s*(?:B|M|K|million|billion|thousand)?",
    "EPS": rf"(?:EPS|Earnings\s+Per\s+Share)[:\s]+({NUMBER_PATTERN})",
    "Cash Flow (Operating)": rf"Cash\s*Flow.*Operating[:\s]+(?:[$€₹])?\s*({NUMBER_PATTERN})\s*(?:B|M|K|million|billion|thousand)?",
    "Cash Flow (Investing)": rf"Cash\s*Flow.*Investing[:\s]+(?:[$€₹])?\s*({NUMBER_PATTERN})\s*(?:B|M|K|million|billion|thousand)?",
    "Cash Flow (Financing)": rf"Cash\s*Flow.*Financing[:\s]+(?:[$€₹])?\s*({NUMBER_PATTERN})\s*(?:B|M|K|million|billion|thousand)?",
    "Total Assets": rf"Total\s+Assets[:\s]+(?:[$€₹])?\s*({NUMBER_PATTERN})\s*(?:B|M|K|million|billion|thousand)?",
    "Total Liabilities": rf"Total\s+Liabilities[:\s]+(?:[$€₹])?\s*({NUMBER_PATTERN})\s*(?:B|M|K|million|billion|thousand)?",
    "Shareholder Equity": rf"(?:Equity|Shareholder\s+Equity)[:\s]+(?:[$€₹])?\s*({NUMBER_PATTERN})\s*(?:B|M|K|million|billion|thousand)?",
}

CONFIDENCE_THRESHOLD = 0.85  # for your entity filtering
