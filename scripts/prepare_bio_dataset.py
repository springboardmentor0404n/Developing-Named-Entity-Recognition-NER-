import os
import json
import random
import re
import nltk
from nltk.corpus import wordnet
from html import unescape
import logging
import spacy
from tqdm import tqdm

# ---------------- CONFIG ----------------
INPUT_FILE = "data/processed/preprocessed_dataset.jsonl"
OUTPUT_FILE = "data/processed/bio_annotation_ready.jsonl"

# Augmentation control
AUGMENT_RATIO = 0.02        # Fraction of records to augment
REPLACE_PROB = 0.10         # per-token synonym replacement
DELETE_PROB = 0.03          # per-token deletion
RANDOM_SEED = 42             # reproducibility

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
random.seed(RANDOM_SEED)

# ---------------- SETUP ----------------
# spaCy tokenizer
try:
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
except OSError:
    logging.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise SystemExit

nlp.max_length = 2000000  # allow large docs

# WordNet setup
try:
    _ = wordnet.synsets("finance")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# Protected financial/numeric tokens
PROTECTED_TOKENS = {
    "₹", "$", "INR", "USD", "EUR", "%", "percent", "percentage",
    "EBITDA", "EBIT", "P/E", "PE", "EPS", "EPS(TTM)",
    "BSE", "NSE", "NASDAQ", "NYSE", "SENSEX", "NIFTY",
    "crore", "lakh", "million", "billion", "trillion"
}

# Regex helpers
RE_NUMERIC = re.compile(r"\d")
RE_XML_TAG = re.compile(r"<[^>]+>")
RE_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# ---------------- HELPERS ----------------
def is_numeric(token: str) -> bool:
    return bool(RE_NUMERIC.search(token)) or token in {"%", "$", "₹", "USD", "INR"}

def clean_text(text: str) -> str:
    """Remove HTML/XML/XBRL and unescape entities."""
    if not text:
        return ""
    text = RE_XML_TAG.sub(" ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_synonym(word: str) -> str:
    if word.upper() in (t.upper() for t in PROTECTED_TOKENS):
        return word
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            cand = lemma.name().replace("_", " ")
            if not re.search(r"\d", cand) and re.match(r"^[A-Za-z\- ]+$", cand) and len(cand) >= 3:
                if cand.lower() != word.lower():
                    syns.add(cand)
    if not syns:
        return word
    single_word_syns = [s for s in syns if " " not in s]
    pool = single_word_syns or list(syns)
    return random.choice(pool)

def tokenize(text: str):
    return RE_TOKEN.findall(text)

def detokenize(tokens: list) -> str:
    out = ""
    for tok in tokens:
        if re.match(r"^[^\w\s]+$", tok):
            out = out.rstrip() + tok + " "
        else:
            out += tok + " "
    return out.strip()

def augment_text(text: str) -> str:
    tokens = tokenize(text)
    augmented = []
    for tok in tokens:
        if is_numeric(tok) or tok.upper() in (t.upper() for t in PROTECTED_TOKENS):
            augmented.append(tok)
            continue
        if re.match(r"^[^\w\s]+$", tok):
            augmented.append(tok)
            continue
        if random.random() < DELETE_PROB:
            continue
        if random.random() < REPLACE_PROB:
            tok = get_synonym(tok)
        augmented.append(tok)
    return detokenize(augmented)

def convert_to_bio(text: str) -> dict:
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_space]
    labels = ["O"] * len(tokens)
    return {"tokens": tokens, "labels": labels, "text": text, "entities": []}

# ---------------- MAIN ----------------
def run_pipeline(input_path=INPUT_FILE, output_path=OUTPUT_FILE):
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total, saved = 0, 0
    seen_texts = set()

    logging.info("Starting full cleaning + deduplication + augmentation + BIO preparation...")

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Processing records"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                raw_text = record.get("text", "").strip()
                if not raw_text:
                    continue
                total += 1

                cleaned_text = clean_text(raw_text)

                # Skip duplicates
                if cleaned_text in seen_texts:
                    continue
                seen_texts.add(cleaned_text)

                # Original cleaned record
                bio_record = convert_to_bio(cleaned_text)
                bio_record["source_file"] = record.get("source_file", "N/A")
                bio_record["augmentation_type"] = "original_clean"
                outfile.write(json.dumps(bio_record) + "\n")
                saved += 1

                # Augmentation (controlled ratio)
                if AUGMENT_RATIO > 0 and random.random() < AUGMENT_RATIO:
                    aug_text = augment_text(cleaned_text)
                    bio_aug_record = convert_to_bio(aug_text)
                    bio_aug_record["source_file"] = record.get("source_file", "N/A")
                    bio_aug_record["augmentation_type"] = "synonym_replace_delete"
                    outfile.write(json.dumps(bio_aug_record) + "\n")
                    saved += 1

            except Exception as e:
                logging.warning(f"Skipping record due to error: {e}")

    logging.info("✅ Pipeline complete.")
    logging.info(f"Original records processed: {total}")
    logging.info(f"Total saved (cleaned + augmented): {saved}")
    logging.info(f"Output → {output_path}")

if __name__ == "__main__":
    run_pipeline()
