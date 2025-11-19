# finbert_eval_v3.py
import os, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR = "./finbert_finetuned_v3"
CSV_PATH = "financeinsight_labeled_with_positive.csv"
SEED = 42

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

# load the same val split you used during training (rebuild split deterministically)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
df = pd.read_csv(CSV_PATH)
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"].str.len() > 0].copy()
df["label"] = df["sentiment"].map(label2id)
df = df[df["label"].isin([0,1,2])].drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

has_company = "company" in df.columns and df["company"].notnull().any()
if has_company:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    groups = df["company"].astype(str)
    idx_train, idx_val = next(gss.split(df, groups=groups))
else:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    idx_train, idx_val = next(sss.split(df, df["label"]))

val_df = df.iloc[idx_val].reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(map(str, texts))
        self.labels = list(map(int, labels))
        self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

val_ds = ValDataset(val_df["clean_text"], val_df["label"], tokenizer)
trainer = Trainer(model=model, tokenizer=tokenizer)
pred = trainer.predict(val_ds)
y_true = val_df["label"].to_numpy()
y_pred = pred.predictions.argmax(-1)

print("\n=== Classification report (val) ===")
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(3)], digits=4))

print("\n=== Confusion matrix (rows=truth, cols=pred) ===")
print(confusion_matrix(y_true, y_pred))
