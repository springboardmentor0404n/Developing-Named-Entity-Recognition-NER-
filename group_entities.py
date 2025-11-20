import re

entity_group_to_rnda = {
    "ORG": "company",
    "MONEY": "value",
    "QUANTITY": "value",
    "DATE": "time",
    "EXCHANGE": "exchange"
}

def normalize_value(text):
    if "billion" in text.lower():
        num = float(re.findall(r"\d+\.?\d*", text)[0])
        return int(num * 1_000_000_000)
    return text

def group_entities(filtered):
    rnda_block = {}
    for ent in filtered:
        if ent["entity_group"] in ["MONEY", "QUANTITY"]:
            ent["normalized_value"] = normalize_value(ent["word"])
        
        rnda_type = entity_group_to_rnda.get(ent["entity_group"], "other")
        if rnda_type not in rnda_block:
            rnda_block[rnda_type] = ent.get("normalized_value", ent["word"])
    
    return rnda_block
