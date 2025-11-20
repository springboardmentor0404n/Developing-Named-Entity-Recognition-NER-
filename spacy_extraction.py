import spacy
import json

nlp = spacy.load("en_core_web_trf")

with open("sample_financial_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

doc = nlp(text)

output = {
    "Organization": [],
    "Financial Amount": [],
    "Reporting Date": [],
    "Percentage": [],
    "Unclassified": []
}

for ent in doc.ents:
    label = ent.label_
    value = ent.text

    if label == "ORG":
        output["Organization"].append(value)
    elif label == "MONEY":
        output["Financial Amount"].append(value)
    elif label == "DATE":
        output["Reporting Date"].append(value)
    elif label == "PERCENT":
        output["Percentage"].append(value)
    else:
        output["Unclassified"].append(value)

with open("rnda_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print("✅ RNDA output saved to rnda_output.json")
