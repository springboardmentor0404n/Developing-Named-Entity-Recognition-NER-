# event_extractor.py
import re

EVENT_PATTERNS = {
    "merger_acquisition": [
        r"\b(acquir(?:ed|es|ing)|acquisition|merge|merger|bought|buyout|takeover)\b",
        r"\b(?:acquired|announced acquisition of)\b"
    ],
    "ipo": [
        r"\b(initial public offering|IPO|went public|priced shares at)\b"
    ],
    "earnings_call": [
        r"\b(earnings call|conference call|quarterly results|report(ed)? (results|earnings))\b"
    ],
    "stock_split": [
        r"\b(stock split|split shares|split of)\b"
    ]
}

def extract_events(text: str):
    found = []
    for evt, patterns in EVENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, text, flags=re.IGNORECASE):
                found.append({"event": evt, "pattern": p, "snippet": text[:240]})
                break
    return found

if __name__ == "__main__":
    txt = "Acme Corp announced an acquisition of ExampleCo. They also announced an earnings call next Tuesday."
    print(extract_events(txt))
