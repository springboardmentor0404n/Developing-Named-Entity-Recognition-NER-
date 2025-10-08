"""
STEP 2 — Basic Text Preprocessing
Clean and normalize unified dataset before tokenization & model training.
"""

import os
import re
import json
import unicodedata
import logging

INPUT_FILE = "data/processed/merged_dataset.jsonl"
OUTPUT_FILE = "data/processed/cleaned_dataset.jsonl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Helper Cleaning Functions

def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFKC form and strip non-printable chars."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch.isprintable())
    return text


def remove_html_tags(text: str) -> str:
    """Remove HTML tags using regex."""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and trim."""
    return " ".join(text.split())


def normalize_currency(text: str) -> str:
    """Replace common currency symbols with consistent tokens."""
    text = re.sub(r"[\$]", " USD ", text)
    text = re.sub(r"[€]", " EUR ", text)
    text = re.sub(r"[₹]", " INR ", text)
    return text


def normalize_dates(text: str) -> str:
    """
    Convert date formats to consistent patterns like YYYY-MM-DD or YYYY-QX.
    Examples:
      Q1 2023 → 2023-Q1
      June 30 2024 → 2024-06-30
    """
    # Quarter formats: Q1 2023 or Q1-2023
    text = re.sub(r"\b(Q[1-4])[\s\-]*(\d{4})\b", r"\2-\1", text)

    # Month Day Year formats
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12"
    }
    month_regex = r"\b(" + "|".join(months.keys()) + r")\s+(\d{1,2}),?\s+(\d{4})"
    def month_repl(m):
        month_num = months[m.group(1).lower()]
        day = int(m.group(2))
        year = m.group(3)
        return f"{year}-{month_num}-{day:02d}"
    text = re.sub(month_regex, month_repl, text, flags=re.IGNORECASE)

    return text


def normalize_abbreviations(text: str) -> str:
    """Standardize spacing around common financial abbreviations."""
    text = re.sub(r"\bP\s*/\s*E\b", "P/E", text, flags=re.IGNORECASE)
    text = re.sub(r"\bE\s*/\s*P\b", "E/P", text, flags=re.IGNORECASE)
    text = re.sub(r"\bE\s*P\s*S\b", "EPS", text, flags=re.IGNORECASE)
    text = re.sub(r"\bE\s*B\s*I\s*T\s*D\s*A\b", "EBITDA", text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    """Apply all cleaning steps in order."""
    text = normalize_unicode(text)
    text = remove_html_tags(text)
    text = normalize_currency(text)
    text = normalize_dates(text)
    text = normalize_abbreviations(text)
    text = normalize_whitespace(text)
    return text.strip()


# Main Processing

def preprocess_dataset(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    logging.info("--- Starting Text Preprocessing ---")

    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleaned_count = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            try:
                record = json.loads(line)
                text = record.get("text", "").strip()
                if not text:
                    continue

                cleaned_text = clean_text(text)
                if not cleaned_text:
                    continue

                # Overwrite text with cleaned version
                record["text"] = cleaned_text
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                cleaned_count += 1

            except json.JSONDecodeError:
                continue

    logging.info(f"✅ Text Preprocessing Complete. {cleaned_count} records cleaned.")
    logging.info(f"Cleaned dataset saved to: {output_file}")


if __name__ == "__main__":
    preprocess_dataset()
