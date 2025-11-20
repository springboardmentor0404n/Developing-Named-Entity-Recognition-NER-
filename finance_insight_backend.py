
# finance_insight_backend.py
# Minimal backend for Milestone 3: NER + Rules + Events + Yahoo verification

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re, calendar
from functools import lru_cache
from statistics import mean


# -------------------- MODEL PATH --------------------
# Update this to your actual local checkpoint folder:
MODEL_PATH = r"D:\Finance-Insight\models\finbert_ner_weighted\checkpoint-9284"

# Globals populated at runtime
model = None
tokenizer = None
id2label: Optional[Dict[int, str]] = None


def set_model(m, tok, id2lbl: Dict[int, str]):
    """Use an already-loaded Hugging Face model/tokenizer/id2label (e.g., from a notebook)."""
    global model, tokenizer, id2label
    model, tokenizer, id2label = m, tok, id2lbl


def set_model_path(path: str):
    """Optionally update the model path at runtime before first load."""
    global MODEL_PATH
    MODEL_PATH = path


def load_model_if_needed():
    """Lazy-load local checkpoint; treats MODEL_PATH as a local folder (not a HF repo id)."""
    global model, tokenizer, id2label
    if model is not None and tokenizer is not None and id2label is not None:
        return

    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()

    # id2label mapping from config (keys can be str in config)
    cfg_map = model.config.id2label
    id2label = {int(k): v for k, v in cfg_map.items()}


def pdf_clean(text: str) -> str:
    # normalize whitespace
    if not text:
        return ""
    t = re.sub(r'[ \t]+', ' ', text)
    t = re.sub(r'\u00A0', ' ', t)  # non-breaking spaces
    t = re.sub(r'\s*\n\s*', '\n', t)

    # drop lines that are mostly numbers/symbols (tables) - conservative
    cleaned_lines = []
    for line in t.splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue
        numy = sum(1 for w in tokens if re.fullmatch(r'[\$€£₹]?[\(\)\-]*\d[\d,\.]*%?', w))
        # drop only if a majority are numeric-like and the line is short (likely a pure table row)
        if len(tokens) <= 8 and (numy / max(1, len(tokens)) > 0.75):
            continue
        cleaned_lines.append(line)
    t = "\n".join(cleaned_lines)

    # collapse multiple newlines
    t = re.sub(r'\n{2,}', '\n', t)
    return t.strip()


# -------------------- FLEXIBLE HELPERS --------------------
SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

MONEY_PATTERN = re.compile(
    r'(?P<cur>[$₹€£]|\bUSD\b|\bINR\b|\bEUR\b|\bGBP\b)?\s*'
    r'(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*'
    r'(?P<unit>trillion|billion|million|thousand|crore|cr|lakh|lac|bn|mn|tn|k)?',
    re.I
)

UNIT_MULT = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6,
    "billion": 1e9, "bn": 1e9,
    "trillion": 1e12, "tn": 1e12,
    "crore": 1e7, "cr": 1e7,
    "lakh": 1e5, "lac": 1e5,
}


def pick_currency(cur: Optional[str]) -> str:
    if not cur:
        return "USD"
    cur = cur.upper()
    return {"$": "USD", "USD": "USD", "₹": "INR", "INR": "INR", "€": "EUR", "EUR": "EUR", "£": "GBP", "GBP": "GBP"}.get(cur, "USD")


def norm_money_match(m: re.Match) -> Dict[str, Any]:
    raw = m.group(0)
    cur = pick_currency(m.group("cur"))
    unit = (m.group("unit") or "").lower()
    num = (m.group("num") or "").replace(",", "")
    try:
        val = float(num)
        # skip numbers that look like years
        if 1900 <= val <= 2100:
            return {}
        val *= UNIT_MULT.get(unit, 1.0)
    except Exception:
        return {"raw": raw}
    return {"raw": raw, "value": val, "unit": unit, "currency": cur}


PERCENT_PATTERN = re.compile(r'(?P<p>\d{1,3}(?:\.\d+)?)\s*%|\b(?P<wordp>\d{1,3}(?:\.\d+)?)\s*percent\b', re.I)


def norm_percent_match(m: re.Match) -> Dict[str, Any]:
    p = m.group("p") or m.group("wordp")
    try:
        val = float(p) / 100.0
    except Exception:
        val = None
    return {"raw": m.group(0), "value": val}


# -------------------- SIMPLE NORMALIZERS (legacy) --------------------
def normalize_money(text: str):
    m = re.search(r"\$?\s?([\d\.]+)\s?(billion|million|trillion|bn|m|t)?", text, re.I)
    if not m:
        return {"raw": text}
    value = float(m.group(1))
    unit = (m.group(2) or "").lower()
    mult = {"billion": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6, "trillion": 1e12, "t": 1e12}.get(unit, 1)
    return {"raw": text, "value": value * mult, "unit": unit if unit else "", "currency": "USD"}


def normalize_percent(text: str):
    m = re.search(r"([\d\.]+)%", text)
    return {"raw": text, "value": float(m.group(1)) / 100 if m else None}


# -------------------- FLEXIBLE RULE-BASED ENTITIES --------------------
def user_defined_extraction(text: str, user_entities: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Flexible extractor:
    - Works even if word order varies ('$3.50 EPS', 'EPS of $3.50', 'reported EPS: $3.50')
    - Supports multiple currencies (USD/INR/EUR/GBP) & Indian units (crore/lakh)
    - Revenue growth picks any % around revenue/sales verbs
    - Stock trend finds verbs + percentage anywhere in the sentence
    """
    results = {ent: [] for ent in user_entities}
    sentences = SENT_SPLIT.split(text.strip()) if text.strip() else []

    for sent in sentences:
        low = sent.lower()

        # --- market_cap ---
        if "market_cap" in user_entities:
            if any(k in low for k in ["market cap", "market capitalization", "valuation"]):
                for m in MONEY_PATTERN.finditer(sent):
                    nm = norm_money_match(m)
                    # keep only valid numeric values (skip year-like matches)
                    if nm and nm.get("value"):
                        results["market_cap"].append(nm)

        # --- EPS ---
        if "EPS" in user_entities:
            if any(k in low for k in ["eps", "earnings per share"]):
                # currency/amount anywhere in the sentence
                for m in MONEY_PATTERN.finditer(sent):
                    nm = norm_money_match(m)
                    if nm and nm.get("value") is not None:
                        results["EPS"].append(nm)
                # numeric EPS without currency: "EPS was 3.50"
                for m in re.finditer(r'\b(?:eps|earnings per share)\b.*?\b(\d+(?:\.\d+)?)\b', sent, re.I):
                    try:
                        num = float(m.group(1))
                        results["EPS"].append({"raw": m.group(0), "value": num, "unit": "", "currency": "USD"})
                    except Exception:
                        pass

        # --- revenue_growth ---
        if "revenue_growth" in user_entities:
            if any(k in low for k in ["revenue", "sales"]):
                if any(v in low for v in ["growth", "grew", "rose", "increased", "decreased", "fell", "declined", "up", "down"]):
                    for m in PERCENT_PATTERN.finditer(sent):
                        results["revenue_growth"].append(norm_percent_match(m))

        # --- stock_price_trend ---
        if "stock_price_trend" in user_entities:
            if any(k in low for k in ["stock", "share price", "stock price", "shares"]):
                if any(v in low for v in ["rose", "fell", "increased", "decreased", "gained", "lost", "up", "down", "jumped", "slumped", "surged"]):
                    for m in PERCENT_PATTERN.finditer(sent):
                        results["stock_price_trend"].append({"raw": m.group(0)})

    # --- de-dup by (raw, value) per entity ---
    for ent, items in results.items():
        seen = set()
        uniq = []
        for it in items:
            key = (it.get("raw"), it.get("value"))
            if key not in seen:
                uniq.append(it)
                seen.add(key)
        results[ent] = uniq

    return results


# -------------------- NER INFERENCE --------------------
def ner_infer(text: str, conf_threshold: float = 0.50):
    """Runs FinBERT NER and returns merged spans with char offsets and scores."""
    load_model_if_needed()
    import torch

    with torch.inference_mode():
        enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        offsets = enc["offset_mapping"][0].tolist()
        logits = model(**{k: v for k, v in enc.items() if k != "offset_mapping"}).logits[0]
        probs = logits.softmax(-1)

    spans = []
    current = None
    for i, (s, e) in enumerate(offsets):
        if s == e:  # special tokens
            continue
        pred_id = int(probs[i].argmax().item())
        label = id2label.get(pred_id, "O")
        score = float(probs[i, pred_id].item())

        if label.startswith("B-") and score >= conf_threshold:
            if current:
                spans.append(current)
            current = {"label": label[2:], "start": s, "end": e, "score": score}
        elif label.startswith("I-") and current and current["label"] == label[2:]:
            current["end"] = e
            current["score"] = max(current["score"], score)
        else:
            if current:
                spans.append(current)
            current = None

    if current:
        spans.append(current)

    for sp in spans:
        try:
            sp["text"] = text[sp["start"]:sp["end"]]
        except Exception:
            sp["text"] = ""

    return spans


# -------------------- NER LABEL → USER ENTITY MAP --------------------
LABEL_TO_ENTITY = {
    "MARKET_CAP": "market_cap",
    "EPS_VALUE": "EPS",
    "REVENUE_GROWTH": "revenue_growth",
    "PRICE_TREND": "stock_price_trend",
    # extend to your model's label set as needed
}


def map_ner_to_user_entities(ner_spans, user_entities):
    out = {e: [] for e in user_entities}
    for sp in ner_spans:
        ent = LABEL_TO_ENTITY.get(sp.get("label"))
        if ent in user_entities:
            out[ent].append({
                "source": "ner",
                "text": sp.get("text"),
                "start": sp.get("start"),
                "end": sp.get("end"),
                "confidence": sp.get("score")
            })
    return out


# -------------------- MERGE (NER + RULES) --------------------
def merge_extractions(text: str, user_entities: List[str], conf_threshold: float = 0.50):
    ner_spans = []
    try:
        ner_spans = ner_infer(text, conf_threshold=conf_threshold)
    except Exception:
        # if model not loaded or fails, continue with rule-based results
        ner_spans = []

    ner_dict = map_ner_to_user_entities(ner_spans, user_entities)
    rule_dict = user_defined_extraction(text, user_entities)

    merged = {e: [] for e in user_entities}
    seen = set()

    def key_of(v):
        if isinstance(v, dict) and "text" in v and v.get("text") is not None:
            return ("ner", v["text"].strip().lower())
        if isinstance(v, dict) and "raw" in v and v.get("raw") is not None:
            return ("rule", v["raw"].strip().lower())
        return ("rule", str(v).strip().lower())

    # prefer model spans first, then rules (dedup by text/raw)
    for e in user_entities:
        for v in ner_dict.get(e, []):
            k = (e, key_of(v))
            if k not in seen:
                merged[e].append(v)
                seen.add(k)
        for v in rule_dict.get(e, []):
            v = {"source": "rule", **v} if isinstance(v, dict) else {"source": "rule", "raw": str(v)}
            k = (e, key_of(v))
            if k not in seen:
                merged[e].append(v)
                seen.add(k)

    return merged


# -------------------- EVENT DETECTION --------------------
_MONTHS = {}
for i in range(1, 13):
    _MONTHS[calendar.month_name[i].lower()] = i
    _MONTHS[calendar.month_abbr[i].lower()] = i


def _parse_date_from_sentence(sent: str) -> Optional[datetime]:
    s = sent.strip()
    # Month [Day], Year  or  Month Year
    m = re.search(
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
        r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})?,?\s+(\d{4})",
        s, re.I
    )
    if m:
        key = m.group(1).lower()
        month = _MONTHS.get(key) or _MONTHS.get(key[:3])
        day = int(m.group(2)) if m.group(2) else 1
        year = int(m.group(3))
        try:
            return datetime(year, month, day)
        except Exception:
            return None

    # Qn YYYY
    m = re.search(r"\bQ([1-4])\s+(\d{4})\b", s, re.I)
    if m:
        q, year = int(m.group(1)), int(m.group(2))
        month = (q - 1) * 3 + 1
        return datetime(year, month, 1)

    # YYYY only
    m = re.search(r"\b(20\d{2}|19\d{2})\b", s)
    if m:
        return datetime(int(m.group(1)), 1, 1)

    return None


_EVENT_KEYWORDS = {
    "M&A": ["merger", "acquisition", "acquire", "merged", "acquired", "buyout"],
    "IPO": ["ipo", "initial public offering", "went public", "listing", "listed on"],
    "stock_split": ["stock split", "share split", "split shares"],
    "earnings_call": ["earnings call", "quarterly results", "earnings report", "conference call"],
    "dividend": ["dividend declared", "dividend announced", "dividend payout", "dividend yield"],
    "guidance": ["guidance", "outlook", "revenue forecast", "earnings guidance"],
    "rating_change": ["credit rating", "downgrade", "upgrade", "rating agency"],
}


def detect_financial_events(
    text: str,
    event_types: List[str],
    timeframe: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Keyword-based event detection; optionally filters by sentence date if found."""
    start, end = timeframe or (None, None)
    results = {et: [] for et in event_types}
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    for sent in sentences:
        low = sent.lower()
        for et in event_types:
            if any(k in low for k in _EVENT_KEYWORDS.get(et, [])):
                d = _parse_date_from_sentence(sent)
                if start and d and d < start:
                    continue
                if end and d and d > end:
                    continue
                results[et].append({"sentence": sent.strip(), "date": d.isoformat() if d else None})
    return results


# ======= YAHOO FINANCE / TICKER VERIFICATION (IMPROVED) =======

from functools import lru_cache
from typing import List, Dict, Any, Optional

# -------------- Simple ticker name -> ticker map (useful for name matching) --------------
ticker_map = {
    "Apple Inc.": "AAPL",
    "Microsoft Corp.": "MSFT",
    "Amazon.com Inc": "AMZN",
    "Alphabet Inc Class A": "GOOGL",
    "Alphabet Inc Class C": "GOOG",
    "Tesla Inc": "TSLA",
    "Johnson & Johnson": "JNJ",
    "Walmart Inc.": "WMT",
    "JPMorgan Chase & Co.": "JPM",
    "Visa Inc.": "V",
    "Procter & Gamble": "PG",
    "Coca-Cola Company (The)": "KO",
    "Berkshire Hathaway": "BRK.B",
    "Home Depot": "HD",
    "Netflix Inc.": "NFLX",
    "Meta Platforms, Inc.": "META",
    "NVIDIA Corporation": "NVDA",
    "Intel Corp.": "INTC",
    "Cisco Systems": "CSCO",
    "Adobe Inc.": "ADBE",
    "Salesforce Inc.": "CRM",
    "Pfizer Inc.": "PFE",
    "Moderna, Inc.": "MRNA",
    "Merck & Co., Inc.": "MRK",
    "AbbVie Inc.": "ABBV",
    "Eli Lilly and Company": "LLY",
    "UnitedHealth Group Inc.": "UNH",
    "McDonald's Corp.": "MCD",
    "Starbucks Corp.": "SBUX",
    "Costco Wholesale Corp.": "COST",
    "Target Corp.": "TGT",
    "Walt Disney Company (The)": "DIS",
    "Boeing Company": "BA",
    "General Electric": "GE",
    "Exxon Mobil Corp.": "XOM",
    "Chevron Corp.": "CVX",
    "AT&T Inc.": "T",
}

# lower-cased name-index for fast lookup
_ticker_name_index = {k.lower(): v for k, v in ticker_map.items()}


def _human_readable_marketcap(market_cap: Optional[float]) -> Optional[str]:
    """Turn a numeric market cap into a short human readable string (e.g., 12.3B)."""
    try:
        if market_cap is None:
            return None
        m = float(market_cap)
        if m >= 1e12:
            return f"{m/1e12:.2f}T"
        if m >= 1e9:
            return f"{m/1e9:.2f}B"
        if m >= 1e6:
            return f"{m/1e6:.2f}M"
        if m >= 1e3:
            return f"{m/1e3:.2f}K"
        return str(int(m))
    except Exception:
        return None


def extract_probable_tickers(text: str) -> List[str]:
    """
    Improved extraction:
      - prefer explicit indications (parentheses, 'ticker:' etc.)
      - then try to match company names via ticker_map (case-insensitive)
      - then fallback to permissive uppercase tokens if nothing found
    Returns unique tickers in appearance order.
    """
    if not text:
        return []

    found = []
    txt = text

    # 1) explicit parentheses: (TSLA)
    for m in re.finditer(r'\((?P<t>[A-Z]{1,6}(?:\.[A-Z])?)\)', txt):
        t = m.group("t").upper()
        if t not in found:
            found.append(t)

    # 2) explicit "ticker: XXX" or "symbol: XXX"
    for m in re.finditer(r'(?:ticker|symbol)\s*[:\-]\s*([A-Z]{1,6}(?:\.[A-Z])?)', txt, re.I):
        t = m.group(1).upper()
        if t not in found:
            found.append(t)

    # 3) company name matches using ticker_map (conservative)
    low = txt.lower()
    # check longest names first to avoid partial matches (sort by length desc)
    for nm in sorted(_ticker_name_index.keys(), key=lambda s: -len(s)):
        if nm in low:
            t = _ticker_name_index[nm]
            if t not in found:
                found.append(t)

    # 4) if still empty, use permissive uppercase fallback but keep small limit
    if not found:
        for m in re.finditer(r'\b([A-Z]{2,5}(?:\.[A-Z])?)\b', txt):
            t = m.group(1).upper()
            if t in {"IPO", "EPS", "PE", "NAV", "ROE", "ROI", "CAGR", "EBITDA", "PV", "IRR", "DCF", "FCF"}:
                continue
            if t not in found:
                found.append(t)
            if len(found) >= 12:
                break

    # final dedupe & return
    return list(dict.fromkeys(found))


@lru_cache(maxsize=512)
def _fetch_ticker_info(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Robust single-ticker fetch using yfinance:
      - tries fast_info, then info, then 1d history for price
      - fetches marketCap, sector/industry if present
      - returns 1y history as small list for trend computations
    Returns None if ticker couldn't be validated (no numeric price).
    """
    try:
        import yfinance as yf
        import pandas as pd
    except Exception:
        return None

    t = ticker.upper()
    try:
        tk = yf.Ticker(t)
    except Exception:
        return None

    out = {"ticker": t, "price": None, "name": None, "market_cap": None, "sector": None, "industry": None, "history": None}

    # Attempt price via multiple paths
    price = None
    try:
        fast = getattr(tk, "fast_info", {}) or {}
        price = fast.get("last_price") or fast.get("lastPrice") or fast.get("previous_close")
    except Exception:
        price = None

    try:
        info = getattr(tk, "info", {}) or {}
        if price is None:
            price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("currentPrice")
    except Exception:
        pass

    # final fallback: 1d history
    if price is None:
        try:
            hist1 = tk.history(period="1d", interval="1d", auto_adjust=False)
            if hist1 is not None and len(hist1) > 0:
                price = float(hist1["Close"].iloc[-1])
        except Exception:
            price = None

    # fill metadata
    try:
        info = getattr(tk, "info", {}) or {}
        out["name"] = info.get("longName") or info.get("shortName") or info.get("symbol") or t
        out["market_cap"] = info.get("marketCap") or info.get("market_cap")
        out["sector"] = info.get("sector")
        out["industry"] = info.get("industry")
    except Exception:
        out["name"] = t

    if price is None:
        return None

    out["price"] = float(price)

    # fetch 1y history (compact)
    try:
        hist = tk.history(period="1y", interval="1d", auto_adjust=True)
        if hist is not None and len(hist) > 10:
            hist_small = [{"date": str(idx.date()), "close": float(row["Close"])} for idx, row in hist.iterrows()]
            out["history"] = hist_small
    except Exception:
        out["history"] = None

    return out


@lru_cache(maxsize=16)
def _fetch_benchmark(bench_symbol: str = "^GSPC") -> Optional[Dict[str, Any]]:
    """Fetch benchmark history (S&P500 default) for relative return calculation."""
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        b = yf.Ticker(bench_symbol)
        hist = b.history(period="1y", interval="1d", auto_adjust=True)
        if hist is None or len(hist) < 10:
            return None
        hist_small = [{"date": str(idx.date()), "close": float(row["Close"])} for idx, row in hist.iterrows()]
        return {"symbol": bench_symbol, "history": hist_small}
    except Exception:
        return None


def _compute_returns_from_history(hist_small: Optional[List[Dict[str, float]]]) -> Dict[str, Optional[float]]:
    """Given small history list compute 1M, 3M, 1Y percent returns."""
    if not hist_small or len(hist_small) < 5:
        return {"1M": None, "3M": None, "1Y": None}
    closes = [x["close"] for x in hist_small]
    n = len(closes)

    def pct_change(from_idx):
        try:
            return (closes[-1] - closes[from_idx]) / closes[from_idx] * 100.0
        except Exception:
            return None

    one_month_idx = max(0, n - 22)
    three_month_idx = max(0, n - 65)
    one_year_idx = 0
    return {"1M": pct_change(one_month_idx), "3M": pct_change(three_month_idx), "1Y": pct_change(one_year_idx)}


def verify_with_financial_db(text: str, include_benchmark: bool = True) -> Dict[str, Any]:
    """
    Main verification:
      - gather ticker candidates via extract_probable_tickers (using ticker_map)
      - fallback to permissive uppercase extraction if none found
      - validate each ticker with _fetch_ticker_info
      - compute simple trend metrics and optionally compare vs benchmark
    """
    verified = {"tickers": []}
    try:
        import yfinance as yf  # quick check
    except Exception:
        return verified

    # 1) candidates from name/ticker heuristics
    cand = extract_probable_tickers(text) or [t.upper() for t in re.findall(r'\b([A-Z]{2,5}(?:\.[A-Z])?)\b', text)][:12]

    cand = list(dict.fromkeys([c.upper() for c in cand]))  # normalize & dedupe

    # fetch benchmark returns once
    benchmark = None
    bench_returns = None
    if include_benchmark:
        benchmark = _fetch_benchmark("^GSPC")
        if benchmark and benchmark.get("history"):
            bench_returns = _compute_returns_from_history(benchmark["history"])

    out = []
    for t in cand:
        info = _fetch_ticker_info(t)
        if not info:
            # try to resolve via ticker_map values (maybe user wrote full company name)
            # e.g., if cand contains a company full name (unlikely here) — skip
            continue

        returns = _compute_returns_from_history(info.get("history")) if info.get("history") else {"1M": None, "3M": None, "1Y": None}

        entry = {
            "ticker": info["ticker"],
            "name": info.get("name"),
            "price": round(float(info["price"]), 3) if info.get("price") is not None else None,
            "market_cap": info.get("market_cap"),
            "market_cap_human": _human_readable_marketcap(info.get("market_cap")),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "returns": returns,
            "1M_return_pct": round(returns.get("1M"), 3) if returns.get("1M") is not None else None,
            "3M_return_pct": round(returns.get("3M"), 3) if returns.get("3M") is not None else None,
            "1Y_return_pct": round(returns.get("1Y"), 3) if returns.get("1Y") is not None else None,
        }

        # vs benchmark
        if bench_returns and entry["returns"].get("1Y") is not None:
            try:
                entry["vs_benchmark_1Y_pct"] = round(entry["returns"]["1Y"] - bench_returns.get("1Y", 0.0), 3)
            except Exception:
                entry["vs_benchmark_1Y_pct"] = None

        out.append(entry)

    verified["tickers"] = out
    if benchmark:
        verified["benchmark"] = {"symbol": benchmark["symbol"], "returns": bench_returns}
    return verified

# ======= END YAHOO FINANCE HELPERS =======

# -------------------- ONE-CALL PIPELINE --------------------
def analyze_text(
    text: str,
    user_entities: List[str],
    event_types: List[str],
    conf_threshold: float = 0.50,
    timeframe: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None
):
    text = pdf_clean(text)
    ents = merge_extractions(text, user_entities, conf_threshold=conf_threshold)
    evts = detect_financial_events(text, event_types, timeframe=timeframe)
    ver = verify_with_financial_db(text)
    return {"entities": ents, "events": evts, "verified": ver}


# ------------------------------------------------------------------
# Additional: Robust table extraction + numeric parsing + shift correction
# (appended to support Milestone 4 features; safe to keep even if not used)
# ------------------------------------------------------------------

# Imports used by the new helpers
try:
    import pdfplumber
    import pandas as pd
except Exception:
    pdfplumber = None
    pd = None


def extract_text_and_tables_from_pdf(path_or_file) -> Tuple[str, List[Tuple[int, Any]]]:
    """
    Safely extract text and tables from a PDF file using pdfplumber.
    Returns (full_text, list_of_(page_number, DataFrame)).
    If pdfplumber/pandas not installed, returns ("", []).
    """
    if pdfplumber is None or pd is None:
        return "", []

    text_pages = []
    tables = []
    with pdfplumber.open(path_or_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            text_pages.append(txt)
            try:
                for tbl in page.extract_tables() or []:
                    if not tbl:
                        continue
                    header = tbl[0]
                    rows = tbl[1:] if len(tbl) > 1 else []
                    ncols = max(1, len(header))
                    cols = [(str(h).strip() if h and str(h).strip() else f"col_{j}") for j, h in enumerate(header)]
                    norm_rows = []
                    for r in rows:
                        r = [None if (c is None or (isinstance(c, str) and not c.strip())) else str(c).strip() for c in r]
                        if len(r) < ncols:
                            r = r + [None] * (ncols - len(r))
                        elif len(r) > ncols:
                            r = r[:ncols]
                        norm_rows.append(r)
                    if norm_rows:
                        df = pd.DataFrame(norm_rows, columns=cols)
                        tables.append((i, df))
            except Exception:
                # ignore extraction errors for robustness
                pass
    full_text = "\n\n".join(text_pages)
    return full_text, tables


def _parse_number_like(s: str) -> Optional[float]:
    """Parse a string that may contain currency/commas/parens/percent and return float or None."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() in {"-", "—", "nan", "none", "na", "n/a"}:
        return None
    if '%' in s:
        try:
            return float(s.replace('%', '').replace(',', '').replace(' ', ''))
        except Exception:
            return None
    s = re.sub(r'\[\d+\]', '', s)
    neg = False
    if s.startswith('(') and s.endswith(')'):
        neg = True
        s = s[1:-1]
    s = re.sub(r'(?<=\d)[\s](?=\d)', '', s)
    s = s.replace('$', '').replace('€', '').replace('£', '').replace('₹', '').replace(',', '').strip()
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return None


def normalize_table_dataframe(df: Any, context_text: str = "", numeric_frac_threshold: float = 0.2) -> Any:
    """
    Convert table DataFrame to numeric where possible (default threshold 20% parsable).
    Keeps textual columns as cleaned strings.
    """
    if pd is None:
        return df
    numeric = df.copy(deep=True)
    ctx = (context_text or "").lower()
    scale = 1.0
    if 'in millions' in ctx or 'amounts in millions' in ctx:
        scale = 1e6
    elif 'in thousands' in ctx or 'amounts in thousands' in ctx:
        scale = 1e3
    elif 'in billions' in ctx:
        scale = 1e9

    for col in numeric.columns:
        col_series = numeric[col].astype(object)
        parsed = col_series.map(lambda x: _parse_number_like(x))
        n_total = len(parsed)
        n_numeric = sum(1 for v in parsed if v is not None)
        if n_total > 0 and (n_numeric / n_total) >= numeric_frac_threshold:
            numeric[col] = parsed.map(lambda v: None if v is None else float(v) * scale)
        else:
            numeric[col] = col_series.map(lambda x: None if x is None or str(x).strip().lower() in {"", "nan", "none"} else str(x).strip())
    numeric = _attempt_shift_correction(numeric)
    return numeric


def _attempt_shift_correction(numeric_df: Any) -> Any:
    """
    Heuristic: if column i is almost empty and column i+1 is numeric, shift some numeric values right.
    Conservative correction to fix common pdf parsing misalignment.
    """
    if pd is None:
        return numeric_df
    df = numeric_df.copy(deep=True)
    ncols = df.shape[1]
    if ncols < 2:
        return df

    def col_numeric_frac(series):
        return sum(1 for v in series if isinstance(v, (int, float))) / max(1, len(series))

    for i in range(ncols - 1):
        frac_i = col_numeric_frac(df.iloc[:, i])
        frac_next = col_numeric_frac(df.iloc[:, i + 1])
        if frac_i < 0.05 and frac_next > 0.25:
            for r in range(df.shape[0]):
                v_i = df.iat[r, i]
                v_next = df.iat[r, i + 1]
                if (v_next is None or v_next == '') and (isinstance(v_i, (int, float)) or (isinstance(v_i, str) and _parse_number_like(v_i) is not None)):
                    df.iat[r, i + 1] = v_i
                    df.iat[r, i] = None
    return df


def filter_tables(raw_tables: List[Tuple[int, Any]],
                  min_rows: int = 2,
                  min_cols: int = 2,
                  min_text_ratio: float = 0.12) -> List[Tuple[int, Any]]:
    """
    Keep only likely-useful tables:
      - at least min_rows x min_cols
      - not pure garbage (text ratio heuristic)
    """
    if pd is None:
        return []
    kept = []
    for page, df in raw_tables:
        if df.shape[0] < min_rows or df.shape[1] < min_cols:
            continue
        total = df.size
        if total == 0:
            continue
        texty = 0
        for cell in df.values.flatten():
            s = str(cell).strip() if cell is not None else ""
            if not s or s in {"-", "—", "nan"}:
                continue
            if re.search(r'[A-Za-z]', s) or re.search(r'[,A-Za-z\$₹£€]', s):
                texty += 1
        if (texty / total) < min_text_ratio:
            if df.shape[0] < 6:
                continue
        kept.append((page, df))
    return kept


def analyze_pdf_file(
    path_or_file,
    user_entities: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    conf_threshold: float = 0.50,
    timeframe: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None
) -> Dict[str, Any]:
    """
    High-level PDF analysis pipeline:
      - extract text + raw tables
      - clean text
      - segment sections
      - run entity extraction (NER + rules) using user_entities
      - detect events using event_types
      - filter, normalize and classify tables
      - verify tickers via yfinance
    Returns same structure as analyze_text plus tables/sections.
    """
    # defaults if caller didn't pass lists
    if user_entities is None:
        user_entities = ["market_cap", "EPS", "revenue_growth"]
    if event_types is None:
        event_types = ["IPO", "M&A", "earnings_call"]

    raw_text, tables = extract_text_and_tables_from_pdf(path_or_file)
    cleaned = pdf_clean(raw_text)
    sections = segment_sections(cleaned)

    # run entity extraction & events on the cleaned text (so PDF path output matches analyze_text)
    ents = merge_extractions(cleaned, user_entities, conf_threshold=conf_threshold)
    evts = detect_financial_events(cleaned, event_types, timeframe=timeframe)

    # tables: filter + normalize + classify
    tables_filtered = filter_tables(tables, min_rows=2, min_cols=2, min_text_ratio=0.12)
    parsed_tables = []
    for page, df in tables_filtered:
        ctx_lines = cleaned.splitlines()
        start_line = max(0, (page - 1) * 8)
        context = "\n".join(ctx_lines[start_line: start_line + 40])
        numeric_df = normalize_table_dataframe(df, context_text=context, numeric_frac_threshold=0.2)
        ttype = guess_table_type(df)
        parsed_tables.append({"page": page, "type": ttype, "raw": df, "numeric": numeric_df})

    # ticker verification (keeps current conservative approach)
    verified = verify_with_financial_db(cleaned)

    return {
        "text": cleaned,
        "sections": sections,
        "entities": ents,
        "events": evts,
        "tables": parsed_tables,
        "verified": verified
    }


# -------------------- Small placeholder helpers --------------------
def segment_sections(text: str) -> Dict[str, str]:
    """
    Simple heuristic section segmentation:
    - looks for common headings and returns snippets
    - This is a lightweight placeholder; replace with your richer segmentation if you have it.
    """
    if not text:
        return {}
    headings = {
        "Executive Summary": ["executive summary", "overview"],
        "MD&A": ["management's discussion", "management discussion and analysis", "md&a", "management discussion"],
        "Financial Statements": ["financial statements", "consolidated statements", "consolidated balance sheet"],
        "Risk Factors": ["risk factors", "risks related to"],
        "Notes": ["notes to consolidated", "notes to the financial statements", "notes"]
    }
    out = {}
    low = text.lower()
    for name, keys in headings.items():
        out[name] = ""
        for k in keys:
            idx = low.find(k)
            if idx != -1:
                snippet = text[idx: idx + 2000]
                out[name] = snippet
                break
    # mark Executive Summary as found if first few lines look like overview
    if not out.get("Executive Summary"):
        out["Executive Summary"] = text[:800] if len(text) > 0 else ""
    return out


def guess_table_type(df: Any ) -> str:
    """
    Very small heuristic to label tables (Balance Sheet, Income, Cash Flow, Other)
    - looks for keywords in header/text.
    """
    if df is None:
        return "Unknown"
    cols_text = " ".join([str(c).lower() for c in df.columns])
    body_text = " ".join([str(x).lower() for x in df.head(5).astype(str).values.flatten()])
    text = cols_text + " " + body_text
    if any(k in text for k in ["balance", "assets", "liabilities", "equity", "total current assets", "total assets"]):
        return "Balance Sheet"
    if any(k in text for k in ["revenue", "sales", "net income", "earnings", "income before"]):
        return "Income Statement"
    if any(k in text for k in ["cash flows", "net cash", "cash and cash equivalents", "operating activities"]):
        return "Cash Flow"
    return "Other"
