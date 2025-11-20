import re

# Keyword to entity type mapping
keyword_map = {
    "revenue": ["REVENUE", "EARNINGS"],
    "stock": ["STOCK_TICKER", "SECURITY_TYPE", "EXCHANGE"],
    "company": ["ORG"],
    "date": ["DATE"],
    "ipo": ["EVENT"]
}

# Regex patterns for fallback matching
regex_patterns = {
    "REVENUE": r"\$\d+(\.\d+)?\s?(million|billion)",
    "EARNINGS": r"(EPS|net income)\s(of)?\s?\$?\d+(\.\d+)?",
    "STOCK_PRICE": r"\$\d+(\.\d+)?\s?(per share)?"
}

def filter_entities(ner_output, user_keywords, raw_text=None):
    target_labels = set()
    for kw in user_keywords:
        target_labels.update(keyword_map.get(kw.lower(), []))

    filtered = [
        {
            "type": ent.get("entity_group", ent.get("entity")),
            "value": ent["word"],
            "context": ent.get("context", "")
        }
        for ent in ner_output
        if ent.get("entity_group") in target_labels or ent.get("word") in target_labels
    ]

    if raw_text:
        for label, pattern in regex_patterns.items():
            matches = re.findall(pattern, raw_text)
            for match in matches:
                filtered.append({
                    "type": label,
                    "value": " ".join(match) if isinstance(match, tuple) else match,
                    "context": "regex match"
                })

    return filtered
