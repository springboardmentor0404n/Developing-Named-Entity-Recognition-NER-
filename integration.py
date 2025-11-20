# integration.py
import yfinance as yf

def enrich_company(org_name: str):
    """
    Simple heuristic: try to map organization name to ticker manually (extendable).
    In real use, maintain a mapping dict or search layer.
    """
    mapping = {
        "Apple": "AAPL", "Tesla": "TSLA", "Amazon": "AMZN",
        "Microsoft": "MSFT", "Alphabet": "GOOGL", "Google": "GOOGL",
        "Meta": "META", "Pfizer": "PFE"
    }
    ticker = mapping.get(org_name)
    if not ticker:
        return {"org": org_name, "enrichment": "unmapped"}
    t = yf.Ticker(ticker)
    info = {}
    try:
        fast = t.fast_info  # faster access
        info["market_cap"] = getattr(fast, "market_cap", None)
        info["year_high"] = getattr(fast, "year_high", None)
        info["year_low"] = getattr(fast, "year_low", None)
    except Exception:
        pass
    return {"org": org_name, "ticker": ticker, "metrics": info}


def enrich_company(name: str) -> dict:
    # Dummy enrichment for now
    return {
        "company": name,
        "sector": "Technology",
        "hq": "Cupertino, CA",
        "market_cap": "$2.8T"
    }
