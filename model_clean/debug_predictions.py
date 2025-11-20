import spacy
import json

nlp = spacy.load(r"D:\FinanceInsight_Dataset\model_clean\model-best")

with open("eval_data.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

for item in eval_data:
    text = item["text"]
    doc = nlp(text)
    print(f"\nText: {text}")
    print("Predicted Entities:")
    for ent in doc.ents:
        print(f"  {ent.text} [{ent.label_}]")
