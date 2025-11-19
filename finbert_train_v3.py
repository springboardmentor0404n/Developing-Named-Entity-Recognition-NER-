# finbert_train_v3.py
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from collections import Counter

SEED = 42
MAX_LEN = 128
MODEL_NAME = "yiyanghkust/finbert-tone"  # keep same model your team uses
CSV_PATH = "financeinsight_labeled_with_positive.csv"
OUT_DIR = "./finbert_finetuned_v3"

np.random.seed(SEED)
torch.manual_seed(SEED)

print("✅ Step 1: Start leakage-safe FinBERT training (v3)")

# ------------------------
# Load & clean data
# ------------------------
df = pd.read_csv(CSV_PATH)

# Ensure required columns exist
needed_cols = {"clean_text", "sentiment"}
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing columns: {missing}. Needed: {needed_cols}")

# Coerce clean_text to string and drop empties
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"].str.len() > 0].copy()

# Map labels to ints
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["sentiment"].map(label2id)

# Drop rows with unmapped labels
df = df[df["label"].isin([0, 1, 2])].copy()

# Optional leakage guard: if you have a company column, we’ll use it, else fallback
has_company = "company" in df.columns

# Drop duplicate texts to avoid trivial memorization
before = len(df)
df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
after = len(df)
print(f"🧹 Deduped texts: removed {before - after} duplicates; kept {after} rows.\n")

print("📊 Label counts (after cleaning):")
print(df["label"].value_counts(), "\n")

# ------------------------
# Train/Val Split (leakage-safe)
# ------------------------
def leakage_safe_split(frame: pd.DataFrame):
    if has_company and frame["company"].notnull().any():
        print("🔒 Using GroupShuffleSplit by 'company' to avoid leakage.\n")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        groups = frame["company"].astype(str)
        idx_train, idx_val = next(gss.split(frame, groups=groups))
    else:
        print("ℹ️ 'company' column missing or empty — falling back to Stratified split on labels.\n")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        idx_train, idx_val = next(sss.split(frame, frame["label"]))
    return frame.iloc[idx_train].reset_index(drop=True), frame.iloc[idx_val].reset_index(drop=True)

train_df, val_df = leakage_safe_split(df)

print("📊 Split sizes:")
print(f"Train: {len(train_df)} | Val: {len(val_df)}\n")

print("📊 Train labels:")
print(train_df["label"].value_counts(), "\n")

print("📊 Val labels:")
print(val_df["label"].value_counts(), "\n")

# ------------------------
# Tokenizer & Model
# ------------------------
# Work around Torch load safety check (your env already has torch>=2.5; this silences strict check)
os.environ["TRANSFORMERS_NO_TORCH_LOAD_CHECK"] = "1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# We try to prefer safetensors if available; if not, this still works with the env var above
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

# ------------------------
# Dataset wrapper (explicit lists -> dict for Trainer)
# ------------------------
class HFDictDataset(TorchDataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = [str(t) for t in texts]
        self.labels = [int(l) for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        # unwrap from size [1, L] to [L]
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        # some FinBERT tokenizers include token_type_ids; add if present
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item

train_dataset = HFDictDataset(
    texts=train_df["clean_text"].tolist(),
    labels=train_df["label"].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN,
)
val_dataset = HFDictDataset(
    texts=val_df["clean_text"].tolist(),
    labels=val_df["label"].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN,
)

# ------------------------
# Class weights (handle imbalance without faking 100% acc)
# ------------------------
label_counts = Counter(train_df["label"].tolist())
num_classes = 3
total = sum(label_counts.values())
# Inverse-frequency weights
weights = []
for c in range(num_classes):
    count_c = label_counts.get(c, 1)
    weights.append(total / (num_classes * count_c))
class_weights = torch.tensor(weights, dtype=torch.float)

print("⚖️ Class weights (train):", {id2label[i]: float(w) for i, w in enumerate(class_weights)}, "\n")

# ------------------------
# Metrics
# ------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

# ------------------------
# Custom Trainer to inject class-weighted loss
# ------------------------
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Move weights to correct device each step
        cw = self.class_weights.to(logits.device)
        loss_fct = CrossEntropyLoss(weight=cw)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ------------------------
# Training config
# ------------------------
os.makedirs(OUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=os.path.join(OUT_DIR, "results"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    num_train_epochs=4,                 # a bit more room to learn with weights
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir=os.path.join(OUT_DIR, "logs"),
    logging_steps=50,
    seed=SEED,
    report_to=[],                       # no wandb needed
)

trainer = WeightedLossTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ------------------------
# Train & Evaluate
# ------------------------
print("🚀 Training started...")
trainer.train()
print("\n✅ Training complete!")

eval_res = trainer.evaluate()
print("\n📈 Validation metrics:")
for k, v in eval_res.items():
    if isinstance(v, (float, int)):
        print(f"  {k}: {v:.4f}")

# ------------------------
# Save model & tokenizer
# ------------------------
save_path = OUT_DIR
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n💾 Fine-tuned model + tokenizer saved to: {save_path}")
