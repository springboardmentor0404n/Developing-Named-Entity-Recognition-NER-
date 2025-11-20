from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load FinBERT tokenizer and model
model_name = "philschmid/bert-base-uncased-finetuned-ner"




tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Sample financial sentence
text = "Amazon issued 500,000 shares of Common Stock listed on Nasdaq under ticker AMZN."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")

# Predict
outputs = model(**inputs)
logits = outputs.logits
predicted_ids = logits.argmax(-1)

# Decode predictions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
labels = [model.config.id2label[i.item()] for i in predicted_ids[0]]

# Print results
for token, label in zip(tokens, labels):
    print(f"{token}: {label}")
