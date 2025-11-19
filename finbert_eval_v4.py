# finbert_eval_v4.py
"""
Evaluation script for finbert_finetuned_v4.
- Cleans dataset (drops NaNs + duplicates), resets index (fixes KeyError).
- Recreates leakage-safe GroupShuffleSplit by 'company' (same splitting strategy as training).
- Loads tokenizer + fine-tuned model from ./finbert_finetuned_v4.
- Runs batched evaluation on GPU if available.
- Prints classification metrics (accuracy / precision / recall / f1 macro & weighted).
"""

import os
import sys
import math
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# --- Config ---
DATA_CSV = "financeinsight_labeled_with_positive.csv"   # ensure this file is in current cwd
MODEL_DIR = "./finbert_finetuned_v4"                    # saved model dir
MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔎 Starting evaluation (device={DEVICE})\n")

# --- Load CSV ---
if not Path(DATA_CSV).exists():
    print(f"ERROR: dataset file not found at {DATA_CSV}")
    sys.exit(1)

df = pd.read_csv(DATA_CSV)
orig_len = len(df)
print(f"🧾 Loaded {orig_len} rows from {DATA_CSV}")

# --- Basic cleaning: require these columns ---
required_cols = ["clean_text", "sentiment", "company"]
for c in required_cols:
    if c not in df.columns:
        print(f"ERROR: required column '{c}' not found in CSV. Columns: {df.columns.tolist()}")
        sys.exit(1)

# Drop rows missing essential fields and duplicates
df = df.dropna(subset=["clean_text", "sentiment", "company"])
df = df.drop_duplicates(subset=["clean_text", "company"])
print(f"🧹 Cleaned data: {orig_len} -> {len(df)} (dropped NaNs and duplicates)")

# --- RESET index to align positional indices with GroupShuffleSplit outputs (fixes KeyError) ---
df = df.reset_index(drop=True)

# --- Map labels to integers (must match training)
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

# Ensure sentiment is string
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
# Filter to only known labels
df = df[df["sentiment"].isin(label2id.keys())].reset_index(drop=True)
df["label"] = df["sentiment"].map(label2id)

print("\n📊 Label counts (after cleaning):")
print(df["label"].value_counts().sort_index().rename(index=id2label))

# --- Create leakage-safe split using 'company' groups (same approach as training) ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.23, random_state=42)  # test_size matches training ~23%
groups = df["company"].values
# returns generator of (train_idx, test_idx)
train_idx, val_idx = next(gss.split(df, groups=groups))

# Use positional selection -- df was reset_index so .iloc will also work; prefer iloc for clarity
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

print(f"\n🔒 Using GroupShuffleSplit by 'company' to avoid leakage.")
print(f"\n📊 Split sizes:\nTrain: {len(train_df)} | Val: {len(val_df)}")
print("\n📊 Val label distribution:")
print(val_df["label"].value_counts().rename(index=id2label))

# --- Load tokenizer + model ---
if not Path(MODEL_DIR).exists():
    print(f"\nERROR: Saved model directory not found at {MODEL_DIR}. Make sure you've saved finetuned model.")
    sys.exit(1)

print("\n🔁 Loading tokenizer and model from:", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, trust_remote_code=True)
model.to(DEVICE)
model.eval()

# --- Tokenize validation texts (batched) ---
def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

val_texts = val_df["clean_text"]
if val_texts.empty:
    print("No validation examples after splitting. Exiting.")
    sys.exit(1)

encodings = tokenize_texts(val_texts, tokenizer, max_length=MAX_LEN)
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
labels = torch.tensor(val_df["label"].values, dtype=torch.long)

dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Inference loop ---
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        b_input_ids, b_attention_mask, b_labels = [t.to(DEVICE) for t in batch]
        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(b_labels.detach().cpu().numpy().tolist())

# --- Metrics ---
acc = accuracy_score(all_labels, all_preds)
prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
rec_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print("\n📈 Validation metrics:")
print(f"  samples: {len(all_labels)}")
print(f"  accuracy: {acc:.4f}")
print(f"  precision (macro): {prec_macro:.4f}")
print(f"  recall (macro): {rec_macro:.4f}")
print(f"  f1 (macro): {f1_macro:.4f}")
print(f"  f1 (weighted): {f1_weighted:.4f}")

print("\n🧾 Classification report (labels -> negative, neutral, positive):\n")
print(classification_report(all_labels, all_preds, target_names=["negative", "neutral", "positive"], zero_division=0))

# Confusion matrix (printed numerically)
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
print("Confusion matrix (rows=true labels, cols=preds) [neg, neu, pos]:")
print(cm)

# Save predictions to CSV so mentor/demo can inspect exact samples
out_df = val_df.copy()
out_df["pred_label"] = all_preds
out_df["pred_sentiment"] = out_df["pred_label"].map(id2label)
out_df["true_sentiment"] = out_df["label"].map(id2label)
out_csv = "finbert_eval_predictions_v4.csv"
out_df.to_csv(out_csv, index=False)
print(f"\n💾 Saved evaluation predictions to: {out_csv}")

print("\n✅ Evaluation complete.")
