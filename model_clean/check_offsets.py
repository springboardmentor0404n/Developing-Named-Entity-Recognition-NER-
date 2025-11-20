import spacy
import json

nlp = spacy.blank("en")

with open("eval_data.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

for item in eval_data:
    text = item["text"]
    doc = nlp(text)
    print(f"\nText: {text}")
    for start, end, label in item["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is None:
            print(f"❌ Misaligned: ({start}, {end}, {label}) → '{text[start:end]}'")
        else:
            print(f"✅ Aligned: {span.text} [{label}]")
