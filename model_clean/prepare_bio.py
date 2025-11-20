from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("Rupesh2/finbert-ner")


with open("train_data.json", "r") as f:
    data = json.load(f)

bio_data = []

for item in data:
    text = item["text"]
    entities = item["entities"]
    entity_map = {tuple(span[:2]): span[2] for span in entities}

    tokens = tokenizer.tokenize(text)
    token_spans = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]

    labels = []
    for span in token_spans:
        label = "O"
        for ent_span, ent_type in entity_map.items():
            if span[0] == ent_span[0] and span[1] == ent_span[1]:
                label = f"B-{ent_type}"
                break
            elif ent_span[0] < span[0] < ent_span[1]:
                label = f"I-{ent_type}"
                break
        labels.append(label)

    bio_data.append({
        "tokens": tokens,
        "labels": labels
    })

with open("model_clean/bio_data.json", "w") as f:
    json.dump(bio_data, f, indent=2)

print("✅ BIO-formatted data saved to model_clean/bio_data.json")
