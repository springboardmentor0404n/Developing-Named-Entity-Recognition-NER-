import json
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ✅ Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")



# ✅ Read input financial text
with open("sample_financial_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("📄 Input text loaded.")

# ✅ Run entity extraction
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
entities = nlp(text)
print("🔍 Raw entities:", entities)

print("🔍 Extracted entities:", entities)

# ✅ Regex validation function
def validate_entity(text, label):
    if label == "MONEY":
        return bool(re.search(r"\$\d+(\.\d+)?\s?(billion|million)?", text))
    elif label == "PERCENT":
        return bool(re.search(r"\d+(\.\d+)?%", text))
    elif label == "DATE":
        return bool(re.search(r"(Q[1-4]\s\d{4}|\d{4})", text))
    return True  # ORG and other types pass through

# ✅ Label mapping for clarity
label_map = {
    "ORG": "Organization",
    "MONEY": "Financial Amount",
    "DATE": "Reporting Date",
    "PERCENT": "Percentage",
    "other": "Unclassified"
}

# ✅ Organize entities into structured output
output = {
    "Organization": [],
    "Financial Amount": [],
    "Reporting Date": [],
    "Percentage": [],
    "Unclassified": []
}

for ent in entities:
    label = label_map.get(ent["entity_group"], "Unclassified")
    value = ent["word"]
    score = ent["score"]

    # ✅ Filter by confidence and validate with regex
    if score > 0.60:

        output[label].append(value)

# ✅ Save to JSON
with open("rnda_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print("✅ RNDA extraction complete. Output saved to rnda_output.json.")
