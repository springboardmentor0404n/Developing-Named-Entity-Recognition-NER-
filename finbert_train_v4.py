# finbert_train_v4.py
import os
import math
import pandas as pd
import numpy as np
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import GroupShuffleSplit
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

print("✅ Start: FinBERT training (v4) with Focal Loss and leakage-safe split")

# -----------------------------
# 1) Load & clean data
# -----------------------------
CSV_PATH = "financeinsight_labeled_with_positive.csv"   # same file you used before
df = pd.read_csv(CSV_PATH)

# Keep only what we need
needed_cols = ["company", "clean_text", "sentiment"]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df = df[needed_cols].dropna(subset=["clean_text", "sentiment"]).copy()

# Drop exact duplicate texts to avoid leakage-y duplicates
before = len(df)
df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
print(f"🧹 Deduped texts: removed {before - len(df)} duplicates; kept {len(df)} rows.\n")

# Map labels (same mapping as v3 so your eval scripts still match)
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].map(label_map)

print("📊 Label counts (after cleaning):")
print(df["label"].value_counts(), "\n")

# -----------------------------
# 2) Leakage-safe split (by company)
# -----------------------------
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
(train_idx, val_idx) = next(gss.split(df, groups=df["company"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df   = df.iloc[val_idx].reset_index(drop=True)

print("🔒 Using GroupShuffleSplit by 'company' to avoid leakage.\n")
print("📊 Split sizes:")
print(f"Train: {len(train_df)} | Val: {len(val_df)}\n")

print("📊 Train labels:")
print(train_df["label"].value_counts(), "\n")

print("📊 Val labels:")
print(val_df["label"].value_counts(), "\n")

# -----------------------------
# 3) Tokenizer & datasets
# -----------------------------
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(df_part: pd.DataFrame) -> Dict[str, Any]:
    texts = df_part["clean_text"].astype(str).tolist()
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    enc["labels"] = torch.tensor(df_part["label"].tolist(), dtype=torch.long)
    return enc

train_enc = tokenize_batch(train_df)
val_enc   = tokenize_batch(val_df)

train_dataset = Dataset.from_dict({k: v.numpy() if isinstance(v, torch.Tensor) else v
                                   for k, v in train_enc.items()})
val_dataset   = Dataset.from_dict({k: v.numpy() if isinstance(v, torch.Tensor) else v
                                   for k, v in val_enc.items()})

# Make sure columns are right for Trainer
train_dataset = train_dataset.with_format("torch")
val_dataset   = val_dataset.with_format("torch")

# -----------------------------
# 4) Focal Loss (with α per class)
# -----------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    alpha: tensor of shape [num_classes] (per-class weighting)
    gamma: focusing parameter
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("alpha", alpha)  # stays on same device as loss module
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B]
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)                  # pt = softmax prob of the true class
        at = self.alpha[targets]             # select α for each target
        loss = at * (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# α weights from inverse class frequency (on TRAIN ONLY)
num_classes = 3
train_counts = train_df["label"].value_counts().reindex(range(num_classes), fill_value=0).to_numpy()
inv = 1.0 / np.clip(train_counts, 1, None)               # avoid div-by-zero
alpha = inv / inv.sum() * num_classes                    # normalize to ~sum=num_classes
alpha = torch.tensor(alpha, dtype=torch.float32)

gamma = 2.0  # common default

pretty_names = {0: "negative", 1: "neutral", 2: "positive"}
alpha_print = {pretty_names[i]: float(alpha[i]) for i in range(num_classes)}
print(f"⚖️ Focal Loss α (per class): {alpha_print} | γ={gamma}\n")

# -----------------------------
# 5) Model & custom Trainer
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id={"negative": 0, "neutral": 1, "positive": 2},
    trust_remote_code=True,
)

class FocalTrainer(Trainer):
    def __init__(self, *args, focal_alpha=None, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Ensure loss module is on same device
        self.focal_loss = self.focal_loss.to(logits.device)
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# -----------------------------
# 6) Training config
# -----------------------------
use_cuda = torch.cuda.is_available()
args = TrainingArguments(
    output_dir="./finbert_results_v4",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,                      # a bit longer; focal often benefits from a few more steps
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=use_cuda,                           # mixed precision on GPU
    logging_steps=50,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
)

# Simple metrics (focus on macro metrics to watch minority classes)
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

trainer = FocalTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    focal_alpha=alpha,
    focal_gamma=gamma,
)

# -----------------------------
# 7) Train & Save
# -----------------------------
print("🚀 Training with Focal Loss started...")
trainer.train()
print("\n✅ Training complete!")

SAVE_DIR = "./finbert_finetuned_v4"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"💾 Saved model + tokenizer to: {SAVE_DIR}")
