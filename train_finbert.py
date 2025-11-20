from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import json

# Load BIO data
with open("model_clean/bio_data.json", "r") as f:
    raw_data = json.load(f)

# Flatten into Hugging Face format
examples = []
for item in raw_data:
    examples.append({
        "tokens": item["tokens"],
        "labels": item["labels"]
    })

# Label mapping
unique_labels = sorted(set(label for ex in examples for label in ex["labels"]))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Convert labels to IDs
for ex in examples:
    ex["label_ids"] = [label2id[label] for label in ex["labels"]]

# Create dataset
dataset = Dataset.from_list([{"tokens": ex["tokens"], "labels": ex["label_ids"]} for ex in examples])

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Rupesh2/finbert-ner")
model = AutoModelForTokenClassification.from_pretrained(
    "yiyanghkust/finbert-pretrain",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)


# Tokenize
def tokenize_and_align_labels(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = example["labels"] + [0] * (128 - len(example["labels"]))
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels)

# Training setup
training_args = TrainingArguments(
    output_dir="model_clean/finbert_output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="model_clean/logs",
    logging_steps=10
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Train
trainer.train()
trainer.save_model("model_clean/finbert_output")

print("✅ FinBERT training complete")
