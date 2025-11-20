from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Rupesh2/finbert-ner")
tokenizer.save_pretrained("model_clean/finbert_output")
print("✅ Tokenizer files saved to model_clean/finbert_output")
