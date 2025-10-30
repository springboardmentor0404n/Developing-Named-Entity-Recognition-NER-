import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import evaluate

# --- Config ---
DATA_DIR = "data/processed/ner_auto_splits"
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 2  
EPOCHS = 3
OUTPUT_DIR = "data/finbert_ner_results" 

# --- Device Check ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load dataset ---
dataset = load_dataset("json", data_files={
    "train": f"{DATA_DIR}/train.jsonl",
    "validation": f"{DATA_DIR}/validation.jsonl",
    "test": f"{DATA_DIR}/test.jsonl"
})

# --- Print dataset sizes ---
print("📊 Dataset sizes:", {k: len(v) for k, v in dataset.items()})

# --- Load label map ---
with open(f"{DATA_DIR}/label_map.json") as f:
    label_info = json.load(f)
label2id = label_info["label2id"]
id2label = {int(k): v for k, v in label_info["id2label"].items()}

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
).to(device)

# --- Tokenize and align labels ---
def tokenize_and_align(batch):
    tokenized = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = [-100 if w is None else label[w] for w in word_ids]
        labels.append(aligned)
    tokenized["labels"] = labels
    return tokenized

dataset = dataset.map(tokenize_and_align, batched=True)
collator = DataCollatorForTokenClassification(tokenizer)

# --- Metric ---
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    pred_labels = [[id2label[p] for (p, l) in zip(pred, label) if l != -100] 
                   for pred, label in zip(predictions, labels)]
    results = metric.compute(predictions=pred_labels, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

# --- Training Arguments ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",  # ✅ save logs inside same folder
    load_best_model_at_end=True,
    report_to="none",  # disable wandb for simplicity
    save_total_limit=1
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# --- Train ---
trainer.train()

# --- Evaluate ---
eval_results = trainer.evaluate(dataset["test"])
print("\n--- Test Set Metrics ---")
print(f"F1-score: {eval_results['eval_f1']:.4f}")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
