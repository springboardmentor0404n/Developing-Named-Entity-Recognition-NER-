import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("💡 FinBERT v2 Evaluation Started...\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}\n")

# Load tokenizer and model
model_path = "./finbert_finetuned_v2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

# Load test data
df = pd.read_csv("financeinsight_labeled_with_positive.csv")

# Clean + prepare
df = df.dropna(subset=["clean_text", "sentiment"])
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].map(label_map)

texts = df["clean_text"].tolist()
true_labels = df["label"].tolist()

# Tokenize
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
).to(device)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

# Classification report
report = classification_report(true_labels, predictions, target_names=label_map.keys(), digits=4)
print("📊 Classification Report:\n")
print(report)

# Save to CSV
report_dict = classification_report(true_labels, predictions, target_names=label_map.keys(), output_dict=True)
pd.DataFrame(report_dict).transpose().to_csv("finbert_v2_classification_report.csv", index=True)
print("\n💾 Saved detailed report to finbert_v2_classification_report.csv")

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.keys(),
            yticklabels=label_map.keys())
plt.title("FinBERT v2 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("finbert_v2_confusion_matrix.png")
plt.close()

print("\n🖼️ Confusion matrix saved as finbert_v2_confusion_matrix.png")
print("\n✅ Evaluation complete!")
