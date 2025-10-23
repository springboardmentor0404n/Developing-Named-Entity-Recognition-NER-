import os
import json
import logging
import spacy
from tqdm import tqdm
from typing import List, Dict, Any

# --- Configuration ---
INPUT_FILE = "data/processed/augmented_dataset.jsonl"
OUTPUT_FILE = "data/processed/bio_annotation_ready.jsonl"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Load spaCy model only for tokenization (disable POS/NER to keep it fast)
try:
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
except OSError:
    logging.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise SystemExit

# --- Main Logic ---

def convert_to_bio_format(text: str) -> Dict[str, Any]:
    """
    Tokenizes text and prepares it for annotation in BIO format.
    Initially assigns all tokens the 'O' (Outside) tag.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_space]
    # Assign 'O' (Outside) tag to every token initially
    labels = ["O"] * len(tokens) 
    
    # Structure needed for annotation platforms/tools
    return {
        "tokens": tokens,
        "labels": labels,
        "text": text,
        # Placeholder for manual label coordinates after annotation is complete
        "entities": [] 
    }

def run_bio_preparation(input_path=INPUT_FILE, output_path=OUTPUT_FILE):
    """Processes the augmented dataset to prepare it for external annotation."""
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_processed = 0

    logging.info(f"Starting BIO format preparation for {os.path.basename(input_path)}")

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in tqdm(infile, desc="Tokenizing and formatting"):
            try:
                record = json.loads(line)
                text = record.get("text", "").strip()
                if not text:
                    continue

                bio_record = convert_to_bio_format(text)
                
                # Copy relevant metadata (source, augmentation type)
                bio_record["source_file"] = record.get("source_file", "N/A")
                bio_record["augmentation_type"] = record.get("augmentation_type", "original")

                outfile.write(json.dumps(bio_record) + "\n")
                total_processed += 1

            except Exception as e:
                logging.error(f"Failed to process record: {e}")

    logging.info(f"✅ BIO Preparation Complete. {total_processed} records saved to {output_path}")

if __name__ == "__main__":
    run_bio_preparation()
