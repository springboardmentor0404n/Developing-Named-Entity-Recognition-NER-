import spacy
from spacy.training import Example
import json

# Load your trained spaCy model
nlp = spacy.load(r"D:\FinanceInsight_Dataset\model_clean\model-best")

# Load evaluation data
with open("eval_data.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

examples = []
for item in eval_data:
    text = item["text"]
    entities = item["entities"]
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, {"entities": entities})
    examples.append(example)

# Evaluate
scores = nlp.evaluate(examples)

# Print results
print("Precision:", round(scores["ents_p"], 2))
print("Recall:", round(scores["ents_r"], 2))
print("F1 Score:", round(scores["ents_f"], 2))
