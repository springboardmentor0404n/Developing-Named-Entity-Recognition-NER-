import os
import json
import re
import logging
from typing import Dict, Any, Generator

# --- Configuration ---
# Input file from the previous unification step
INPUT_FILE = "data/processed/merged_financial_dataset.jsonl"
# Output file for the cleaned, preprocessed text
OUTPUT_FILE = "data/processed/domain_preprocessed_data.jsonl"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --- Domain-Specific Normalization Rules ---

def normalize_financial_text(text: str) -> str:
    """
    Applies aggressive, domain-specific cleaning and normalization rules
    to standardize financial jargon, dates, and numbers.
    """
    # 1. HTML Tag and Noise Removal (Safety net, though unification should handle most)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Date Normalization (Crucial for NER consistency)
    # Convert 'Q1 2023' or '2023 Q1' style to a standard 'YYYY-QX' format.
    text = re.sub(r'(\d{4})\s*Q([1-4])', r'\1-Q\2', text, flags=re.IGNORECASE)
    text = re.sub(r'Q([1-4])\s*(\d{4})', r'\2-Q\1', text, flags=re.IGNORECASE)
    
    # Attempt to standardize month/day/year formats to YYYY-MM-DD for easier parsing later.
    # This is complex, but we prioritize standardizing separators/order.
    # Example: June 30, 2024 -> 2024-06-30 (Requires a dedicated library like dateutil for robust parsing)
    # For now, let's standardize common separators and spell out the number.
    text = text.replace(',', ' ').replace('/', ' ').replace('-', ' ') 

    # 3. Currency and Symbol Normalization
    # Replace symbols with standardized tokens to treat them as single entities later.
    text = re.sub(r'(\s)[\$€£¥₹](\s)', r'\1CURRENCY_SYMBOL\2', text)
    text = re.sub(r'(\d)(\s)[\$€£¥₹](\s)', r'\1 CURRENCY_SYMBOL\2', text) # Catch cases like "50$"

    # --- ADVANCED OPTIMIZATIONS ---
    # 4a. Standardize Large Number Suffixes (e.g., $5.2B -> $5.2 Billion)
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(B|Bil)\b', r'\1 BILLION', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(M|Mil)\b', r'\1 MILLION', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(K|k)\b', r'\1 THOUSAND', text, flags=re.IGNORECASE)

    # 4b. Standardize 'Per' Mentions (e.g., 'per share', 'per cent')
    text = re.sub(r'\bper\s+share\b', ' PER_SHARE ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bper\s+annum\b', ' PER_ANNUM ', text, flags=re.IGNORECASE)
    # --- END ADVANCED OPTIMIZATIONS ---

    # 5. Abbreviation Standardization (Crucial for transfer learning/model focus)
    # We pad abbreviations with spaces to clearly delineate them as single tokens.
    text = re.sub(r'\b(P/E|PE)\b', ' P_E_RATIO ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(EPS)\b', ' EARNINGS_PER_SHARE ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(EBITDA)\b', ' EBITDA_METRIC ', text, flags=re.IGNORECASE)
    
    # 6. Percentage and Number Cleaning
    # Ensure numbers and percentages have clean spacing
    text = re.sub(r'(\d)\s*%', r'\1 PERCENT_SYMBOL', text)

    # 7. Final Whitespace Cleanup
    text = " ".join(text.split()).strip()
    return text

# --- Pipeline Function (no change) ---

def run_text_preprocessing(input_file: str, output_file: str):
    """
    Loads unified data, applies domain-specific normalization, and saves the cleaned records.
    """
    logging.info(f"Loading unified data from: {input_file}")
    
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}. Please run prepare_dataset.py first.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    total_processed = 0

    with open(input_file, 'r', encoding='utf-8') as in_f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        for line in in_f:
            try:
                record = json.loads(line)
                raw_text = record.get("text", "")
                source_file = record.get("source_file", "N/A")
                
                # Apply the specific financial text transformations
                clean_text = normalize_financial_text(raw_text)
                
                if clean_text:
                    # Preserve the original structure but update the text field
                    record["text"] = clean_text
                    out_f.write(json.dumps(record) + "\n")
                    total_processed += 1
                    
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON line in input file.")
                
    logging.info(f"✅ Text Preprocessing Complete.")
    logging.info(f"   {total_processed} records saved to {output_file}")


if __name__ == "__main__":
    run_text_preprocessing(INPUT_FILE, OUTPUT_FILE)
