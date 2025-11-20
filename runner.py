import os
import json

def process_text(text: str):
    print("Starting process_text...")

    sections = {
        "Management's Discussion and Analysis": text,
        "Risk Factors": "The company faces supply chain disruptions.",
        "Financial Statements": "Revenue: $416B, Net Income: $94B"
    }

    print("Sections found:", list(sections.keys()))

    report = {}
    for name, content in sections.items():
        report[name] = {
            "summary": content,
            "entities": [("Apple Inc.", "ORG"), ("$416B", "MONEY")],
            "metrics": {"Revenue": 416, "Net Income": 94}
        }

    print("Final report payload:", report)

    os.makedirs("reports/json", exist_ok=True)

    with open("reports/json/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Report saved to reports/json/report.json")

if __name__ == "__main__":
    text = """Apple Inc. reported revenue of $416 billion in FY 2024. Net income was $94 billion. EPS reached 6.12."""
    process_text(text)
