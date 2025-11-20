import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import json

nlp = spacy.blank("en")  # create blank English pipeline
db = DocBin()

with open("train_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    text = item["text"]
    entities = item["entities"]
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label)
        if span is None:
            print(f"Skipping misaligned entity: {text[start:end]} → {label}")
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk("train.spacy")
print("✅ Saved as train.spacy")
