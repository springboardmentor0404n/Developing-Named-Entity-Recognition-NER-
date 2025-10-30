import json, os, random
from sklearn.model_selection import train_test_split

AUTO_FILE = "data/processed/bio_auto_annotated.jsonl"
OUTDIR = "data/processed/ner_auto_splits"
TEST_RATIO = 0.1
VAL_RATIO = 0.1
random.seed(42)

os.makedirs(OUTDIR, exist_ok=True)

# Load auto records
records = [json.loads(line) for line in open(AUTO_FILE, encoding="utf-8")]

# Split train / val / test
train_records, temp = train_test_split(records, test_size=VAL_RATIO + TEST_RATIO, random_state=42)
val_records, test_records = train_test_split(temp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), random_state=42)

# Get label map
labels = sorted({lab for r in records for lab in r["labels"]})
if "O" not in labels:
    labels.insert(0, "O")
label2id = {lab: i for i, lab in enumerate(labels)}

def convert(rec):
    return {
        "tokens": rec["tokens"],
        "ner_tags": [label2id.get(l, 0) for l in rec["labels"]],
        "text": rec.get("text", "")
    }

for name, data in [("train", train_records), ("validation", val_records), ("test", test_records)]:
    with open(os.path.join(OUTDIR, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(convert(r)) + "\n")

with open(os.path.join(OUTDIR, "label_map.json"), "w") as f:
    json.dump({"label2id": label2id, "id2label": {v: k for k, v in label2id.items()}}, f, indent=2)


print(f"[INFO] Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
print("[INFO] Label map saved.")
