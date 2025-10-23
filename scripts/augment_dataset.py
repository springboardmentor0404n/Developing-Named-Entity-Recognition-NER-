import os
import json
import random
import re
import nltk
from nltk.corpus import wordnet
import logging

# ---------------- CONFIG ----------------
INPUT_FILE = "data/processed/merged_dataset.jsonl"
OUTPUT_FILE = "data/processed/augmented_dataset.jsonl"
AUGMENT_FACTOR = 1   # number of augmented copies per record

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------- SETUP ----------------
# Download WordNet once if needed
try:
    _ = wordnet.synsets("finance")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

PROTECTED_FINANCIAL_TOKENS = {
    "CURRENCY_SYMBOL", "PERCENT_SYMBOL", "P_E_RATIO", "EARNINGS_PER_SHARE",
    "EBITDA_METRIC", "BILLION", "MILLION", "THOUSAND", "PER_SHARE", "PER_ANNUM"
}

# ---------------- HELPERS ----------------
def is_numeric_token(token: str) -> bool:
    """Detect if token is a number or contains digits (like 2025, 12.3%, $200)."""
    return bool(re.search(r'\d', token))

def get_synonym(word: str) -> str:
    """Return one random synonym for the word, or the same word if none found."""
    if word.upper() in PROTECTED_FINANCIAL_TOKENS:
        return word
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                synonyms.add(candidate)
    synonyms = [s for s in synonyms if not is_numeric_token(s) and s.isalpha()]
    return random.choice(synonyms) if synonyms else word

def augment_text(text: str, replace_prob=0.1, delete_prob=0.05) -> str:
    """
    Simple synonym replacement + random deletion.
    Keeps numbers and protected tokens unchanged.
    """
    words = text.split()
    augmented = []

    for w in words:
        # Skip numeric or protected tokens
        if w.upper() in PROTECTED_FINANCIAL_TOKENS or is_numeric_token(w):
            augmented.append(w)
            continue

        # Random deletion
        if random.random() < delete_prob:
            continue

        # Random synonym replacement
        if random.random() < replace_prob:
            new_w = get_synonym(w)
            augmented.append(new_w)
        else:
            augmented.append(w)

    return " ".join(augmented)

# ---------------- MAIN ----------------
def run_augmentation(input_path=INPUT_FILE, output_path=OUTPUT_FILE, factor=AUGMENT_FACTOR):
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total, saved = 0, 0

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("text", "").strip()
            if not text:
                continue

            # Write original record
            outfile.write(json.dumps(record) + "\n")
            saved += 1

            # Generate augmented copies
            for i in range(factor):
                new_text = augment_text(text)
                aug_record = {
                    "text": new_text,
                    "source_file": record.get("source_file", ""),
                    "augmentation_type": "synonym_replace_delete"
                }
                outfile.write(json.dumps(aug_record) + "\n")
                saved += 1

            total += 1

    logging.info(f"✅ Augmentation done.")
    logging.info(f"   Original records processed: {total}")
    logging.info(f"   Total saved (original + augmented): {saved}")
    logging.info(f"   Output → {output_path}")

if __name__ == "__main__":
    run_augmentation()
