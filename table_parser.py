# table_parser.py
import re
import pandas as pd
import numpy as np
from typing import Optional
from config import ROW_PATTERNS, NUMBER_PATTERN, NULL_TOKENS

def clean_text_table(raw: str) -> str:
    raw = re.sub(r"\f|\r", "\n", raw)            # page breaks
    raw = re.sub(r"\n{2,}", "\n", raw)           # collapse blanks
    raw = re.sub(r"Page\s+\d+|Table\s+\d+", "", raw, flags=re.IGNORECASE)
    for tok in NULL_TOKENS:
        raw = raw.replace(tok, "")
    raw = re.sub(r"[ \t]+", " ", raw)            # normalize spaces
    return raw

def parse_text_table(raw: str) -> pd.DataFrame:
    """
    Parse semi-structured text blocks into Metric/Value pairs using ROW_PATTERNS.
    Fallback to colon-delimited parsing if patterns fail.
    """
    raw = clean_text_table(raw)
    rows = []

    # Pattern-based extraction for known metrics
    for key, pattern in ROW_PATTERNS.items():
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            try:
                rows.append([key, float(val)])
            except ValueError:
                rows.append([key, val])

    # Fallback: generic colon-delimited lines with numeric right side
    if not rows:
        for line in raw.splitlines():
            parts = [p.strip() for p in line.split(":")]
            if len(parts) == 2 and re.search(NUMBER_PATTERN, parts[1]):
                num = re.search(NUMBER_PATTERN, parts[1]).group(0).replace(",", "")
                try:
                    rows.append([parts[0], float(num)])
                except ValueError:
                    rows.append([parts[0], num])

    df = pd.DataFrame(rows, columns=["Metric", "Value"]) if rows else pd.DataFrame(columns=["Metric", "Value"])
    if not df.empty:
        df["Value"] = df["Value"].replace({np.nan: None})
    return df

def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to coerce 'Value' to numeric when possible."""
    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df



import pandas as pd
import re

def parse_text_table(text: str) -> pd.DataFrame:
    lines = [line for line in text.splitlines() if ':' in line]
    rows = [line.split(':') for line in lines]
    return pd.DataFrame(rows, columns=["Metric", "Value"])

def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    def clean(val):
        val = val.replace("$", "").replace("B", "").strip()
        try:
            return float(val)
        except:
            return None
    df["Value"] = df["Value"].apply(clean)
    return df
