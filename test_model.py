from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Test sentence
text = "Apple Inc. reported $89.5 billion in revenue in Q4 2023, with a 15% increase in iPhone sales."

# Run model
entities = nlp(text)

# Print results
for ent in entities:
    print(f"{ent['word']} → {ent['entity_group']} ({ent['score']:.2f})")
