import os
import pandas as pd
import torch
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

print("✅ Step 1: FinBERT Training Pipeline Started...")

# ------------------------------
# 1️⃣ Load dataset
# ------------------------------
df = pd.read_csv("financeinsight_labeled_with_positive.csv")

print("\n📊 Original class distribution:")
print(df["sentiment"].value_counts())

# ------------------------------
# 2️⃣ Map text labels to numeric and balance dataset
# ------------------------------
label_map = {"positive": 1, "negative": 0, "neutral": 2}
df["label"] = df["sentiment"].map(label_map)

# Split by label
df_majority = df[df["label"] == 1]  # Positive
df_minority_neg = df[df["label"] == 0]  # Negative
df_minority_neu = df[df["label"] == 2]  # Neutral

# Oversample minority classes to match majority
df_minority_neg_upsampled = resample(
    df_minority_neg,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_minority_neu_upsampled = resample(
    df_minority_neu,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

# Combine all classes and shuffle
df_balanced = pd.concat([df_majority, df_minority_neg_upsampled, df_minority_neu_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n✅ After balancing:")
print(df_balanced["label"].value_counts())

# Optional: Save balanced dataset
df_balanced.to_csv("financeinsight_balanced.csv", index=False)
print("\n💾 Balanced dataset saved as financeinsight_balanced.csv")

# ------------------------------
# 3️⃣ Prepare columns and clean data
# ------------------------------
df_balanced.dropna(subset=["clean_text", "sentiment"], inplace=True)
df_balanced = df_balanced[["clean_text", "sentiment"]]

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}
df_balanced["label"] = df_balanced["sentiment"].map(label2id)

print("\n✅ Labels mapped successfully!")
print(df_balanced["label"].value_counts())

# ------------------------------
# 4️⃣ Split train/validation
# ------------------------------
train_df, val_df = train_test_split(
    df_balanced,
    test_size=0.2,
    random_state=42,
    stratify=df_balanced["label"]
)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ------------------------------
# 5️⃣ Load FinBERT model & tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
os.environ["TRANSFORMERS_NO_TORCH_LOAD_CHECK"] = "1"

model = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone",
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True
)

# ------------------------------
# 6️⃣ Tokenize data
# ------------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["clean_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ------------------------------
# 7️⃣ Training setup
# ------------------------------
training_args = TrainingArguments(
    output_dir="./finbert_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# ------------------------------
# 8️⃣ Start training
# ------------------------------
print("\n🚀 Training started...")
trainer.train()
print("\n✅ Training complete!")

# ------------------------------
# 9️⃣ Save fine-tuned model
# ------------------------------
trainer.save_model("./finbert_finetuned")
print("\n💾 Fine-tuned model saved to ./finbert_finetuned")

