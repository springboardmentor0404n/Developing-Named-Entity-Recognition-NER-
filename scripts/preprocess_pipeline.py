"""
Preprocessing Pipeline for Financial Text

Stage 1: Clean structure (HTML, spaces, unicode, dates, symbols)
Stage 2: Apply financial-domain specific normalization
"""

import os
import re
import json
import logging
from typing import Generator
 
INPUT_FILE = "data/processed/merged_dataset.jsonl"
OUTPUT_FILE = "data/processed/preprocessed_dataset.jsonl"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# STAGE 1 — STRUCTURAL CLEANING

def clean_structural(text: str) -> str:
    """Basic cleaning: remove HTML, fix spaces, normalize unicode & dates."""
    if not isinstance(text, str):
        text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize unicode characters
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # Normalize whitespace
    text = " ".join(text.split())

    # Standardize date formats like Q1 2023 → 2023-Q1
    text = re.sub(r"(\d{4})\s*Q([1-4])", r"\1-Q\2", text, flags=re.IGNORECASE)
    text = re.sub(r"Q([1-4])\s*(\d{4})", r"\2-Q\1", text, flags=re.IGNORECASE)

    # Remove extra symbols (except finance-relevant ones)
    text = re.sub(r"[^\w\s\.\,\$\€\£\¥\₹\%\-/]", " ", text)

    return text.strip()


# STAGE 2 — FINANCIAL DOMAIN NORMALIZER

def normalize_financial(text: str) -> str:
    """Standardize finance-specific patterns and abbreviations."""

    # Currency symbols → CURRENCY_SYMBOL
    text = re.sub(r"[\$€£¥₹]", " CURRENCY_SYMBOL ", text)

    # Standardize large number suffixes
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[Bb](?:illion)?", r"\1 BILLION", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[Mm](?:illion)?", r"\1 MILLION", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[Kk]", r"\1 THOUSAND", text)

    # Replace % with PERCENT_SYMBOL
    text = re.sub(r"(\d)\s*%", r"\1 PERCENT_SYMBOL", text)

    # Standardize common finance abbreviations
    text = re.sub(r"\bP\/E\b|\bPE\b", " P_E_RATIO ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bEPS\b", " EARNINGS_PER_SHARE ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bEBITDA\b", " EBITDA_METRIC ", text, flags=re.IGNORECASE)

    # Clean up extra whitespace
    text = " ".join(text.split())

    return text.strip()


# MAIN PIPELINE FUNCTION

def preprocess_pipeline(input_path: str = INPUT_FILE, output_path: str = OUTPUT_FILE):
    """Runs Stage 1 + Stage 2 preprocessing and saves cleaned text."""
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_records = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            try:
                record = json.loads(line)
                raw_text = record.get("text", "")
                if not raw_text:
                    continue

                # Apply Cleaning Pipeline
                text = clean_structural(raw_text)
                text = normalize_financial(text)

                if text:
                    record["text"] = text
                    outfile.write(json.dumps(record) + "\n")
                    total_records += 1

            except json.JSONDecodeError:
                logging.warning("Skipping malformed JSON line.")

    logging.info(f" Preprocessing Complete! {total_records} records saved to {output_path}")


if __name__ == "__main__":
    preprocess_pipeline()
