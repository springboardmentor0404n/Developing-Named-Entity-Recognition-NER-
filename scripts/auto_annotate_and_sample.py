# scripts/auto_annotate_and_sample.py
import os
import json
import re
import logging
import random
from collections import Counter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
random.seed(42)

INPUT = "data/processed/bio_annotation_ready.jsonl"
OUTPUT = "data/processed/bio_auto_annotated.jsonl"
SAMPLE_DIR = "data/processed/manual_sample"
SAMPLE_FILE = os.path.join(SAMPLE_DIR, "manual_validation_sample.jsonl")
SAMPLE_SIZE = 300  # change 200-500 as you prefer

# --- Load spaCy model (prefer transformer if available) ---
try:
    import spacy
except ImportError:
    logging.error("spaCy not installed. Run: pip install spacy")
    raise SystemExit

MODEL_NAMES = ["en_core_web_trf", "en_core_web_sm"]
nlp = None
for m in MODEL_NAMES:
    try:
        nlp = spacy.load(m)
        logging.info(f"Loaded spaCy model: {m}")
        break
    except Exception:
        continue

if nlp is None:
    logging.error("No spaCy model found. Install one with `python -m spacy download en_core_web_sm`")
    raise SystemExit

# --- Simple rule regexes ---
RE_MONEY = re.compile(r'\b(?:₹|Rs\.?|USD|US\$|\$|EUR|£|GBP|INR)\s?[\d\.,]+(?:\s?(?:crore|lakh|million|billion|bn|m))?\b', re.IGNORECASE)
RE_PERCENT = re.compile(r'\b\d+(?:[\.,]\d+)?\s?(?:%|percent|pct)\b', re.IGNORECASE)
RE_QUARTER = re.compile(r'\bQ[1-4](?:\s*\d{2,4})?\b', re.IGNORECASE)
RE_YEAR = re.compile(r'\b(?:19|20)\d{2}\b')

# Map spaCy entity labels -> our entity labels (adjust if needed)
SPACY_TO_ENTITY = {
    "ORG": "ORG",
    "DATE": "DATE",
    "MONEY": "FIN_VALUE",
    "PERCENT": "FIN_VALUE",
    "PERSON": None,
    "GPE": "ORG",     # map geo-entities to ORG (optional)
    "PRODUCT": "FIN_TERM",
    # add more mappings if you want
}

def find_rule_spans(text):
    spans = []
    for m in RE_MONEY.finditer(text):
        spans.append((m.start(), m.end(), "FIN_VALUE"))
    for m in RE_PERCENT.finditer(text):
        spans.append((m.start(), m.end(), "FIN_VALUE"))
    for m in RE_QUARTER.finditer(text):
        spans.append((m.start(), m.end(), "DATE"))
    for m in RE_YEAR.finditer(text):
        spans.append((m.start(), m.end(), "DATE"))
    return spans

def merge_spans(spans):
    """Merge overlapping spans; prefer longer spans when overlapping."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], -(x[1]-x[0])))
    merged = []
    for s,e,label in spans:
        if not merged:
            merged.append([s,e,label])
            continue
        ms, me, ml = merged[-1]
        if s <= me:  # overlap
            if e > me:
                merged[-1][1] = e
                merged[-1][2] = label if (e-s) > (me-ms) else ml
        else:
            merged.append([s,e,label])
    return [(a,b,c) for a,b,c in merged]

def spans_to_bio(doc, spans):
    tokens = [t.text for t in doc]
    labels = ["O"] * len(tokens)
    for (s,e,label) in spans:
        # find token indexes that cover this char span
        start_i = None
        end_i = None
        for i, tok in enumerate(doc):
            if start_i is None and tok.idx <= s < tok.idx + len(tok.text):
                start_i = i
            if start_i is not None and tok.idx < e <= tok.idx + len(tok.text):
                end_i = i
                break
        # fallback heuristic if not found
        if start_i is None or end_i is None:
            for i, tok in enumerate(doc):
                if tok.idx >= s:
                    start_i = i
                    break
            if start_i is None:
                continue
            for j in range(start_i, len(doc)):
                if doc[j].idx + len(doc[j].text) >= e:
                    end_i = j
                    break
            if end_i is None:
                end_i = start_i
        # assign BIO
        labels[start_i] = f"B-{label}"
        for k in range(start_i+1, end_i+1):
            labels[k] = f"I-{label}"
    return tokens, labels

def auto_annotate_record(text):
    doc = nlp(text)
    spans = []
    # model spans
    for ent in doc.ents:
        mapped = SPACY_TO_ENTITY.get(ent.label_, None)
        if mapped:
            spans.append((ent.start_char, ent.end_char, mapped))
    # rule spans
    spans.extend(find_rule_spans(text))
    # merge and convert
    merged = merge_spans(spans)
    tokens, labels = spans_to_bio(doc, merged)
    return tokens, labels, merged

def main():
    if not os.path.exists(INPUT):
        logging.error("Input file not found: " + INPUT)
        return

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    out_f = open(OUTPUT, "w", encoding="utf-8")
    sample_candidates = []
    stats = Counter()

    total = 0
    skipped = 0
    for line in open(INPUT, encoding="utf-8"):
        total += 1
        try:
            rec = json.loads(line)
            text = rec.get("text", "") or " ".join(rec.get("tokens", []))
            tokens, labels, spans = auto_annotate_record(text)
            # Save stats
            for _,_,lab in spans:
                stats[lab] += 1
            out_rec = {
                "tokens": tokens,
                "labels": labels,
                "text": text,
                "source_file": rec.get("source_file", ""),
                "augmentation_type": rec.get("augmentation_type", "auto_annotated")
            }
            out_f.write(json.dumps(out_rec) + "\n")
            # if this record has any non-O labels, add to candidate pool for manual sampling
            if any(l != "O" for l in labels):
                sample_candidates.append(out_rec)
        except Exception as e:
            skipped += 1
            logging.warning(f"Skipped record {total} due to error: {e}")
            continue

    out_f.close()

    # Save a sample for manual correction (most useful: records that have at least one entity)
    sample_count = min(SAMPLE_SIZE, len(sample_candidates))
    if sample_count > 0:
        sample = random.sample(sample_candidates, sample_count)
        with open(SAMPLE_FILE, "w", encoding="utf-8") as sf:
            for r in sample:
                sf.write(json.dumps(r) + "\n")
        logging.info(f"Saved manual review sample ({sample_count}) -> {SAMPLE_FILE}")
    else:
        logging.info("No auto-labeled records found to sample.")

    # Print brief stats
    logging.info(f"Processed: {total}, Skipped: {skipped}")
    logging.info("Auto-label counts by entity:")
    for k,v in stats.most_common():
        logging.info(f"  {k}: {v}")

    logging.info("Auto-annotated file written to: " + OUTPUT)
    logging.info("Manual sample (correct these) written to: " + SAMPLE_FILE)

if __name__ == "__main__":
    main()
