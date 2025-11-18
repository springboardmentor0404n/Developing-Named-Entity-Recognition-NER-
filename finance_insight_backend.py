
# finance_insight_backend.py
# Minimal backend for Milestone 3: NER + Rules + Events + Yahoo verification

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re, calendar

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


# -------------------- TICKERS + YAHOO FINANCE --------------------
def extract_tickers(text: str) -> List[str]:
    """
    Extract probable tickers even if not inside parentheses or labelled.
    """
    STOP = {"IPO", "EPS", "PE", "NAV", "ROE", "ROI", "CAGR", "EBITDA", "PV", "IRR", "DCF", "FCF"}
    seen, out = set(), []
    if not text:
        return out
    for m in re.finditer(r'\b([A-Z]{1,5}(?:\.[A-Z])?)\b', text):
        t = m.group(1)
        if t in STOP or len(t) < 2:
            continue
        # small whitelist to help detect real tickers
        if t in {"AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NFLX"} or "(" in text or "ticker" in text.lower():
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def extract_probable_tickers(text: str) -> List[str]:
    """Conservative probable ticker extraction (parentheses or explicit 'ticker' hints)."""
    found = []
    if not text:
        return []
    for m in re.finditer(r'\((?P<t>[A-Z]{1,5}(?:\.[A-Z])?)\)', text):
        found.append(m.group("t"))
    for m in re.finditer(r'(?:ticker|symbol)\s*[:\-]\s*([A-Z]{1,5}(?:\.[A-Z])?)', text, re.I):
        found.append(m.group(1).upper())
    # simple name→ticker guesses
    low = text.lower()
    if "apple" in low and "AAPL" not in found:
        found.append("AAPL")
    if "tesla" in low and "TSLA" not in found:
        found.append("TSLA")
    return list(dict.fromkeys(found))


def verify_with_financial_db(text: str) -> Dict[str, Any]:
    """Verifies/enriches tickers using yfinance (name + last price)."""
    verified = {"tickers": []}
    try:
        import yfinance as yf
    except Exception:
        return verified

    tickers = extract_probable_tickers(text) or extract_tickers(text)
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            price = info.get("regularMarketPrice")
            name = info.get("longName") or info.get("shortName")
            if price is not None:
                verified["tickers"].append({"ticker": t, "price": price, "name": name or "N/A"})
        except Exception:
            # ignore invalid tickers / network errors
            pass
    return verified


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


def extract_text_and_tables_from_pdf(path_or_file) -> Tuple[str, List[Tuple[int, 'pd.DataFrame']]]:
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


def normalize_table_dataframe(df: 'pd.DataFrame', context_text: str = "", numeric_frac_threshold: float = 0.2) -> 'pd.DataFrame':
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


def _attempt_shift_correction(numeric_df: 'pd.DataFrame') -> 'pd.DataFrame':
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


def filter_tables(raw_tables: List[Tuple[int, 'pd.DataFrame']],
                  min_rows: int = 2,
                  min_cols: int = 2,
                  min_text_ratio: float = 0.12) -> List[Tuple[int, 'pd.DataFrame']]:
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


def analyze_pdf_file(path_or_file) -> Dict[str, Any]:
    """
    High-level PDF analysis pipeline:
      - extract text + raw tables
      - clean text
      - segment sections (simple heuristic placeholder)
      - filter, normalize and classify tables
      - verify tickers via yfinance
    """
    raw_text, tables = extract_text_and_tables_from_pdf(path_or_file)
    cleaned = pdf_clean(raw_text)
    sections = segment_sections(cleaned)  # uses placeholder below
    tables_filtered = filter_tables(tables, min_rows=2, min_cols=2, min_text_ratio=0.12)
    parsed_tables = []
    for page, df in tables_filtered:
        ctx_lines = cleaned.splitlines()
        # small context snippet based on page index (approximation)
        start_line = max(0, (page - 1) * 8)
        context = "\n".join(ctx_lines[start_line: start_line + 40])
        numeric_df = normalize_table_dataframe(df, context_text=context, numeric_frac_threshold=0.2)
        ttype = guess_table_type(df)  # placeholder below
        parsed_tables.append({"page": page, "type": ttype, "raw": df, "numeric": numeric_df})
    tickers = extract_probable_tickers(cleaned)
    verified = verify_with_financial_db(cleaned)
    return {"text": cleaned, "sections": sections, "tables": parsed_tables, "verified": verified}


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


def guess_table_type(df: 'pd.DataFrame') -> str:
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
