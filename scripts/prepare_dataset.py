"""
Unify raw financial datasets (csv, json, txt, pdf, html) into one clean JSONL file.
- Removes duplicates
- Normalizes whitespace
"""

import os
import json
import hashlib
import unicodedata
import logging
import pandas as pd
from bs4 import BeautifulSoup

# Try importing PDF libraries
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# --- Paths ---
RAW_DIR = "data/raw"
OUTPUT_FILE = "data/processed/merged_dataset.jsonl"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Helper Functions

def normalize_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch.isprintable())
    text = " ".join(text.split())
    return text.strip()


def sha256_hash(text: str) -> str:
    """Short hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Extractors

def extract_txt(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if content:
            yield content
    except Exception as e:
        logging.error(f"TXT extraction failed for {os.path.basename(path)}: {e}")
    

def extract_html(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            if text:
                yield text
    except Exception as e:
        logging.error(f"HTML extraction failed for {os.path.basename(path)}: {e}")
            

def extract_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            # Fallback
            df = pd.read_csv(path, encoding="ISO-8859-1", on_bad_lines="skip")
        except Exception as e:
            logging.error(f"CSV read failed for {path} due to encoding: {e}")
            return    
        
    text_keywords = ["text", "sentence", "content", "description", "headline", "body","transcript"]
    text_cols = [c for c in df.columns if any(k in c.lower() for k in text_keywords)]

    if not text_cols and not df.empty:
        text_cols = [df.columns[0]]
    for col in text_cols:
        for val in df[col].dropna():
            t = str(val).strip()
            if t:
                yield t


def extract_json(path):
    """Handles JSON and JSONL formats."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data_loaded = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                for key in ["text", "sentence", "content", "body"]:
                    if key in obj and obj[key]:
                        yield str(obj[key]).strip()
                        break
            except json.JSONDecodeError:
                continue

        if not data_loaded:
                f.seek(0)
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            for key in ["text", "sentence", "content", "body"]:
                                if key in item and item[key]:
                                    yield str(item[key]).strip()
                                    break
                    elif isinstance(data, dict):
                        for key in ["text", "sentence", "content", "body"]:
                            if key in data and data[key]:
                                yield str(data[key]).strip()
                                break
                except Exception as e:
                    logging.error(f"JSON object parsing failed for {path}: {e}")


def extract_pdf(path):
    """PDF → text using pdfplumber first, fallback to pypdf."""
    if pdfplumber:
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text(x_tolerance=1)
                    if t:
                        yield t.strip()
            return
        except Exception as e:
            logging.warning(f"pdfplumber failed for {path}.  Attempting pypdf fallback. Error: {e}")
    if PdfReader:
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    yield t.strip()
        except Exception as e:
            logging.error(f"PDF fallback failed for {path}: {e}")

# Main Processing

EXT_HANDLERS = {
    ".txt": extract_txt,
    ".text": extract_txt,
    ".html": extract_html,
    ".htm": extract_html,
    ".csv": extract_csv,
    ".json": extract_json,
    ".jsonl": extract_json,
    ".pdf": extract_pdf
}


def prepare_dataset(raw_dir=RAW_DIR, output_file=OUTPUT_FILE, enable_deduplication=True):
    logging.info("--- Starting Financial Text Dataset Unification ---")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    seen_hashes = set()
    total_records = 0
    total_deduplicated = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for root, _, files in os.walk(raw_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                handler = EXT_HANDLERS.get(ext)
                if not handler:
                    logging.debug(f"Skipping unsupported file: {fname}")
                    continue

                path = os.path.join(root, fname)
                logging.info(f"Processing {fname}")

                for raw_text in handler(path):
                    text = normalize_text(raw_text)
                    if not text:
                        continue
                    if enable_deduplication:
                        current_hash = sha256_hash(text)
                        if current_hash in seen_hashes:
                            total_deduplicated += 1
                            continue
                        seen_hashes.add(current_hash)

                    out_f.write(json.dumps({"text": text, "source_file": fname}) + "\n")
                    total_records += 1    

    logging.info(f"✅ Finished! Total {total_records} unique records saved to {output_file} ")
    if enable_deduplication:
        logging.info(f"Total duplicates removed: {total_deduplicated}")

if __name__ == "__main__":
    prepare_dataset()
