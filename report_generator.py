# report_generator.py
import json
from typing import Dict, List
import pandas as pd

def section_payload(section_name: str, df: pd.DataFrame, ents: List[tuple], enrichments: List[dict]) -> dict:
    # Build concise, human-readable lines from DF
    lines = []
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            lines.append(f"{row['Metric']}: {row['Value']}")
    ent_list = [{"text": t, "label": l} for t, l in ents]
    payload = {
        "section": section_name,
        "table_metrics": df.to_dict(orient="records") if df is not None else [],
        "entities": ent_list,
        "enrichment": enrichments,
        "summary": "; ".join(lines)
    }
    return payload

def write_json_report(report: Dict, out_path: str):
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)


import json

def section_payload(name, df, ents, enrichments):
    return {
        "section": name,
        "entities": ents,
        "table_metrics": df.to_dict(orient="records"),
        "enrichment": enrichments,
        "summary": f"{name} contains {len(ents)} entities and {len(df)} metrics."
    }

def write_json_report(report, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
