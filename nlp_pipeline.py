# nlp_pipeline.py
import spacy

def load_nlp():
    # SpaCy transformer pipeline leveraging Hugging Face under the hood
    nlp = spacy.load("en_core_web_trf")
    return nlp

def extract_entities(text, nlp):
    doc = nlp(text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    return ents


import spacy

def load_nlp():
    return spacy.load("en_core_web_sm")

def extract_entities(text: str, nlp) -> list:
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
