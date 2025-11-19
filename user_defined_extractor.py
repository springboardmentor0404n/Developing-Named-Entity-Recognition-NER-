# user_defined_extractor.py
import re
from typing import List, Dict, Any

# Example patterns for common financial entities (adjust/add rules)
ENTITY_PATTERNS = {
    "EPS": [
        r"\b(EPS|earnings per share)\b[:\s]*([-+]?\d+\.?\d*)",
        r"\bearnings(?:\s+per\s+share)?\s+(?:of\s+)?\$?([-+]?\d+\.?\d*)"
    ],
    "market_cap": [
        r"\bmarket cap(?:italization)?\b[:\s]*\$?([\d,.]+(?:\s*(?:billion|million|bn|m|B|M))?)",
    ],
    "revenue_growth": [
        r"\b(revenue|sales)\s+(?:growth|increased|grew|up)\s+(?:by\s+)?([-\d\.]+%)",
    ],
    "eps_change": [
        r"\b(EPS|earnings per share)\s+(?:increased|decreased|fell|rose)\s+by\s+([-\d\.]+%)",
    ],
    # ticker pattern (simple)
    "ticker": [
        r"\b([A-Z]{1,5})\b(?=\s+(?:shares|stock|ticker|closed|rose|fell|jumped))",
        r"\b([A-Z]{1,5})\b(?=\s*[:\-]\s*stock)"
    ]
}

def extract_entities(text: str, user_entities: List[str]=None):
    user_entities = user_entities or list(ENTITY_PATTERNS.keys())
    findings = []
    for ent in user_entities:
        patterns = ENTITY_PATTERNS.get(ent, [])
        for p in patterns:
            for m in re.finditer(p, text, flags=re.IGNORECASE):
                findings.append({
                    "entity": ent,
                    "match": m.group(0),
                    "groups": m.groups(),
                    "start": m.start(),
                    "end": m.end(),
                    "context": text[max(0, m.start()-80): m.end()+80]
                })
    return findings

# quick test runner
if __name__ == "__main__":
    sample = "EPS was $1.23 this quarter. Market cap: $12.3 billion. Sales growth up 5.2% year-on-year. AAPL shares rose."
    print(extract_entities(sample, ["EPS", "market_cap", "revenue_growth", "ticker"]))
