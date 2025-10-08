"""
FinanceInsight: Linguistic Feature Extraction
---------------------------------------------
Stage 3 - Tokenization, POS Tagging, and Lemmatization

This script processes cleaned financial text to extract linguistic features
used for EDA, data cleaning, and NER model training.
"""

import os
import json
import spacy
import logging
import pandas as pd
from tqdm import tqdm

# Paths
INPUT_FILE = "data/processed/preprocessed_dataset.jsonl"
OUTPUT_FILE = "data/processed/linguistic_features.jsonl"
STATS_FILE = "data/processed/token_stats.csv"

# Logging Setup 
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Load spaCy Model
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    logging.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise SystemExit


# Linguistic Feature Extraction

def extract_features(text: str):
    """Tokenize text and extract POS, lemmas, and content lemmas."""
    doc = nlp(text)
    tokens, pos_tags, lemmas, content_lemmas = [], [], [], []

    for token in doc:
        if token.is_space:
            continue
        tokens.append(token.text)
        pos_tags.append(token.pos_)
        lemma = token.lemma_.lower()
        lemmas.append(lemma)

        if not token.is_stop and not token.is_punct and token.text.strip():
            content_lemmas.append(lemma)

    return {
        "tokens": tokens,
        "pos_tags": pos_tags,
        "lemmas": lemmas,
        "content_lemmas": content_lemmas,
        "token_count": len(tokens),
        "unique_token_count": len(set(lemmas)),
    }



# Main Pipeline Function

def process_dataset(input_path=INPUT_FILE, output_path=OUTPUT_FILE, stats_path=STATS_FILE):
    """
    Processes a JSONL dataset line-by-line to extract linguistic features.
    Saves both enriched JSONL data and token statistics for EDA.
    """
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total, token_counts = 0, []

    logging.info(f"Starting linguistic feature extraction from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile, desc="Processing records"):
            try:
                record = json.loads(line)
                text = record.get("text", "").strip()
                if not text:
                    continue

                features = extract_features(text)
                record.update(features)

                outfile.write(json.dumps(record) + "\n")
                token_counts.append(features["token_count"])
                total += 1

            except json.JSONDecodeError:
                logging.warning("Skipping malformed JSON line.")
            except Exception as e:
                logging.error(f"Failed to process record: {e}")

    # Save token statistics
    df_stats = pd.DataFrame(token_counts, columns=["token_count"])
    df_stats["length_category"] = pd.cut(
        df_stats["token_count"],
        bins=[0, 50, 100, 250, 500, 1000, 2000],
        labels=["<50", "50–100", "100–250", "250–500", "500–1000", ">1000"]
    )
    df_stats.to_csv(stats_path, index=False)

    logging.info(f"✅ Completed! {total} records processed.")
    logging.info(f"🗂️ Output saved to: {output_path}")
    logging.info(f"📊 Token stats saved to: {stats_path}")


if __name__ == "__main__":
    process_dataset()
