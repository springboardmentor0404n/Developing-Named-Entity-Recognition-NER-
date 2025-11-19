import os
import pandas as pd
import torch
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset
import numpy as np

print("✅ Step 1: FinBERT Training v2 Pipeline Started...")

# -----------------------------
# 1️⃣ Load and Prepare Dataset
# -----------------------------
df = pd.read_csv("financeinsight_labeled_with_positive.csv")

print("\n📊 Original Class Distribution:")
print(df["sentiment"].value_counts())

# Label mapping
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].map(label_map)

# Drop missing clean_text
df = df.dropna(subset=["clean_text"])

# -----------------------------
# 2️⃣ Balance the dataset
# -----------------------------
df_majority = df[df["label"] == df["label"].value_counts().idxmax()]
balanced_frames = []

for lbl in df["label"].unique():
    df_lbl = df[df["label"] == lbl]
    df_bal = resample(df_lbl, replace=True, n_samples=len(df_majority), random_state=42)
    balanced_frames.append(df_bal)

df_balanced = pd.concat(balanced_frames).sample(frac=1, random_state=42).reset_index(drop=True)

print("\n✅ After Balancing:")
print(df_balanced["label"].value_counts())

# Save balanced version
df_balanced.to_csv("financeinsight_balanced.csv", index=False)
print("\n💾 Balanced dataset saved as financeinsight_balanced.csv")

# -----------------------------
# 3️⃣ Tokenization
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

def tokenize_function(examples):
    return tokenizer(examples["clean_text"], truncation=True, padding="max_length", max_length=256)

train_df, val_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced["label"])
train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------------
# 4️⃣ Load Model
# -----------------------------
os.environ["TRANSFORMERS_NO_TORCH_LOAD_CHECK"] = "1"
model = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone",
    num_labels=3,
    trust_remote_code=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\n🧠 Using device: {device}")

# -----------------------------
# 5️⃣ Compute Class Weights
# -----------------------------
label_counts = df_balanced["label"].value_counts().sort_index().values
class_weights = torch.tensor(len(df_balanced) / (len(label_counts) * label_counts), dtype=torch.float32).to(device)
print("\n⚖️  Class Weights:", class_weights)

# Custom loss function (weighted cross-entropy)
from torch.nn import CrossEntropyLoss

def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")
    loss_fct = CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

# -----------------------------
# 6️⃣ Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./finbert_results_v2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir="./logs_v2",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    fp16=True if torch.cuda.is_available() else False,
)

# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------------
# 7️⃣ Define Metrics
# -----------------------------
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# -----------------------------
# 8️⃣ Trainer Setup
# -----------------------------
# ✅ Custom Trainer subclass to apply weighted loss
from transformers import Trainer

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Initialize Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -----------------------------
# 9️⃣ Train
# -----------------------------
print("\n🚀 Training started...\n")
trainer.train()
print("\n✅ Training complete!")

# -----------------------------
# 🔟 Save Model
# -----------------------------
trainer.save_model("./finbert_finetuned_v2")
print("\n💾 Fine-tuned model saved to ./finbert_finetuned_v2")
