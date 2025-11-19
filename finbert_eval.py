import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("\nFinBERT Evaluation Started...\n")

# ------------------------------
# 1️⃣ Load fine-tuned model & tokenizer
# ------------------------------
model_path = "./finbert_finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Using device: {device}")

# ------------------------------
# 2️⃣ Load test dataset (balanced version)
# ------------------------------
df = pd.read_csv("financeinsight_balanced.csv")

# Encode labels
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["sentiment"].map(label2id)

texts = df["clean_text"].tolist()
true_labels = df["label"].tolist()

# ------------------------------
# 3️⃣ Tokenize and predict
# ------------------------------
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

# ------------------------------
# 4️⃣ Evaluation metrics
# ------------------------------
print("\n📊 Classification Report:\n")
print(classification_report(true_labels, preds, target_names=label2id.keys()))

# ------------------------------
# 5️⃣ Confusion Matrix (visual)
# ------------------------------
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label2id.keys(),
            yticklabels=label2id.keys())
plt.title("FinBERT Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("finbert_confusion_matrix.png")
plt.show()

print("\n✅ Confusion matrix saved as finbert_confusion_matrix.png")
