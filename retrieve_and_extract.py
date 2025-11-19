#!/usr/bin/env python3
"""
retrieve_and_extract.py  (fixed build_or_load_index signature)

Single-file end-to-end retrieval + financial entity extraction.
See earlier notes in the project for usage examples.
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss not available. Install faiss-cpu (or faiss-gpu) into your conda env.") from e

import yfinance as yf
import requests

ALPHA_KEY = os.environ.get("ALPHA_VANTAGE_KEY") or os.environ.get("ALPHA_VANTAGE_API_KEY")

CSV_PATH = "financeinsight_labeled_with_positive.csv"
EMB_FILE = "embeddings.npy"
FAISS_FILE = "faiss_index.idx"
META_FILE = "chunks_meta.json"
EXTRA_OUT_DIR = "retrieve_outputs"

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH = 256

EPS_PATTERNS = [
    r"(?i)\bEPS[:\s\-]*([€$¥£]?\s*[-+]?\d{1,3}(?:[,\d{3}]*)(?:\.\d+)?(?:\s*(?:billion|bn|million|m|k|thousand|usd|eur|gbp|usd)?)?)\b",
    r"(?i)\bearnings per share[:\s\-]*([€$¥£]?\s*[-+]?\d{1,3}(?:[,\d{3}]*)(?:\.\d+)?(?:\s*(?:billion|bn|million|m|k|usd|eur)?)?)\b",
    r"(?i)\b(EPS|eps|E\.P\.S\.|earnings per share)\b[^\d]{0,8}([€$¥£]?\s*[-+]?\d{1,3}(?:[,\d{3}]*)(?:\.\d+)?)",
    r"(?i)([-+]?\d{1,3}(?:[,\d{3}]*)(?:\.\d+)?)(?:\s*(USD|usd|EUR|eur|GBP|gbp))\s*(?:\bEPS\b|\bearnings per share\b)?"
]

NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:[,\d{3}]*)(?:\.\d+)?")
MARKETCAP_PATTERNS = [
    r"(?i)\bmarket cap(?:italization)?[:\s]*([€$¥£]?\s*[-+]?\d{1,3}(?:[,\d{3}]*)?(?:\.\d+)?\s*(?:billion|bn|million|m|k|thousand)?)",
    r"(?i)\b(?:market value|market capitalization)[:\s]*([€$¥£]?\s*[-+]?\d+(?:\.\d+)?\s*(?:billion|million|bn|m|k)?)",
    r"(?i)\b([€$¥£]\s*\d{1,3}(?:[,\d{3}]*)?(?:\.\d+)?)\s*(?:market cap|market capitalization)?"
]
REVENUE_PATTERNS = [
    r"(?i)\brevenue(?:s)?\b.*?(?:grew|increased|decreased|fell|rose|up|down|was|is|at)\s*([+-]?\d{1,3}(?:[,\d{3}]*)?(?:\.\d+)?%?)",
    r"(?i)\brevenue(?:s)?[:\s]*([€$¥£]?\s*[-+]?\d+(?:[,\d{3}]*)?(?:\.\d+)?\s*(?:billion|million|m|k)?)"
]

TICKER_RE = re.compile(r"\b\(?\$?([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\)?\b")
UNIT_MAP = {"billion": 1_000_000_000, "bn": 1_000_000_000, "million": 1_000_000, "m": 1_000_000, "k": 1_000}
os.makedirs(EXTRA_OUT_DIR, exist_ok=True)

# ---------- helpers ----------

def load_csv(path=CSV_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found at {path}.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "nan", "None"])
    for col in ["clean_text", "text", "content", "body"]:
        if col in df.columns:
            df["text_clean"] = df[col].astype(str)
            break
    else:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if len(str_cols) == 0:
            raise ValueError("No text-like column found in CSV.")
        df["text_clean"] = df[str_cols[0]].astype(str)
    if "company" not in df.columns:
        df["company"] = df.get("company", "")
    return df.reset_index(drop=True)

def build_embeddings(texts, model_name=EMB_MODEL_NAME, batch_size=BATCH):
    print("Computing embeddings and building FAISS index (this may take a moment)...")
    model = SentenceTransformer(model_name)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")
    return embeddings

def build_or_load_index(rebuild=False):
    # load df and texts internally
    df = load_csv()
    texts = df["text_clean"].astype(str).tolist()
    if (not rebuild) and os.path.exists(EMB_FILE) and os.path.exists(FAISS_FILE) and os.path.exists(META_FILE):
        try:
            print("💾 Loaded embeddings from", EMB_FILE)
            emb = np.load(EMB_FILE)
            index = faiss.read_index(FAISS_FILE)
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return index, emb, meta, df
        except Exception as e:
            print("Failed to load existing index (will rebuild):", e)
    emb = build_embeddings(texts)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(emb)
    index.add(emb)
    np.save(EMB_FILE, emb)
    faiss.write_index(index, FAISS_FILE)
    meta = {"n": len(texts)}
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print("💾 Saved embeddings to", EMB_FILE)
    print("💾 Saved FAISS index to", FAISS_FILE)
    return index, emb, meta, df

def search_index(query, index, k=5, model_name=EMB_MODEL_NAME):
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return I[0].tolist(), D[0].tolist()

def parse_number_with_unit(raw):
    if raw is None:
        return None
    s = raw.strip()
    cur_m = re.search(r"(?P<cur>[$€¥£])", s)
    cur = cur_m.group("cur") if cur_m else ""
    unit = ""
    for k in UNIT_MAP:
        if re.search(r"\b"+re.escape(k)+r"\b", s, flags=re.I):
            unit = k
            break
    num_m = re.search(NUMBER_RE, s.replace(",", ""))
    if not num_m:
        try:
            val = float(s.replace(",", "").split()[0])
            return val * (UNIT_MAP.get(unit, 1)), unit, cur
        except Exception:
            return None
    try:
        val = float(num_m.group(0).replace(",", ""))
    except Exception:
        return None
    val_scaled = val * (UNIT_MAP.get(unit, 1))
    return val_scaled, unit, cur

def find_eps_in_text(text):
    eps_values = []
    eps_currency = []
    eps_trend = []
    for pat in EPS_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            groups = [g for g in m.groups() if g]
            if not groups:
                continue
            candidate = groups[-1].strip()
            parsed = parse_number_with_unit(candidate)
            if parsed:
                val, unit, cur = parsed
                eps_values.append(val)
                eps_currency.append(cur)
            nearby = m.group(0)
            if re.search(r"\b(increase|increased|up|rose|grew|gain)\b", nearby, flags=re.I):
                eps_trend.append("increased")
            if re.search(r"\b(decrease|decreased|fell|down|drop|loss)\b", nearby, flags=re.I):
                eps_trend.append("decreased")
    if re.search(r"(?i)earnings per share.*\b(increased|decreased|grew|fell|rose|up|down)\b", text):
        m = re.search(r"(?i)earnings per share.*\b(increased|decreased|grew|fell|rose|up|down)\b", text)
        if m:
            eps_trend.append(m.group(1).lower())
    return list(dict.fromkeys(eps_values)), list(dict.fromkeys(eps_trend)), list(dict.fromkeys(eps_currency))

def find_marketcap_in_text(text):
    found = []
    for pat in MARKETCAP_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            g = m.group(1) if m.groups() else m.group(0)
            if not g:
                continue
            p = parse_number_with_unit(g)
            if p:
                val, unit, cur = p
                found.append({"raw": g.strip(), "value": val, "unit": unit, "currency": cur})
    return found

def find_revenue_mentions(text):
    mentions = []
    for pat in REVENUE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            mentions.append(m.group(0).strip())
    return list(dict.fromkeys(mentions))

def extract_tickers(text):
    cand = []
    for m in TICKER_RE.finditer(text):
        g = m.group(1)
        if g and re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z]{1,2})?", g):
            cand.append(g)
    return list(dict.fromkeys(cand))

def yfinance_eps_fallback(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        trailing = info.get("trailingEps")
        forward = info.get("forwardEps")
        return {"ticker": ticker, "trailingEps": trailing, "forwardEps": forward, "shortName": info.get("shortName")}
    except Exception as e:
        print("yfinance lookup failed:", e)
        return None

def alpha_symbol_search(name, api_key=ALPHA_KEY):
    if not api_key:
        return []
    url = "https://www.alphavantage.co/query"
    params = {"function": "SYMBOL_SEARCH", "keywords": name, "apikey": api_key}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        matches = data.get("bestMatches", [])
        symbols = [m.get("1. symbol") for m in matches if m.get("1. symbol")]
        return symbols
    except Exception as e:
        print("AlphaVantage search error:", e)
        return []

def yahoo_search_suggestions(name, max_retries=2):
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": name, "quotesCount": 6, "newsCount": 0}
        for attempt in range(max_retries):
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                j = r.json()
                quotes = j.get("quotes", [])
                symbols = [q.get("symbol") for q in quotes if q.get("symbol")]
                return symbols
            if r.status_code == 429:
                time.sleep(1 + 2 * attempt)
            else:
                break
    except Exception:
        pass
    return []

def extract_entities(text):
    eps_vals, eps_trend, eps_currency = find_eps_in_text(text)
    market_caps = find_marketcap_in_text(text)
    revenue_mentions = find_revenue_mentions(text)
    tickers_raw = extract_tickers(text)

    tickers_valid = []
    for t in tickers_raw:
        try:
            info = (yf.Ticker(t).info or {})
            if info.get("shortName") or info.get("regularMarketPrice") is not None:
                tickers_valid.append(t)
        except Exception:
            pass

    return {
        "eps": eps_vals,
        "eps_trend": eps_trend,
        "eps_currency": eps_currency,
        "market_cap": market_caps,
        "revenue_mention": revenue_mentions,
        "tickers_raw": tickers_raw,
        "tickers_valid": tickers_valid,
        "raw_text": text
    }

def canonical_eps_from_results(results, fallback_ticker=None):
    eps_candidates = []
    for r in results:
        for v in r.get("eps", []):
            try:
                eps_candidates.append(float(v))
            except Exception:
                pass
    if eps_candidates:
        val = float(np.median(eps_candidates))
        return val, "source=doc"
    trends = []
    for r in results:
        trends.extend(r.get("eps_trend", []))
    if trends:
        return None, f"trend={'|'.join(list(dict.fromkeys(trends)))}"
    if fallback_ticker:
        yfdata = yfinance_eps_fallback(fallback_ticker)
        if yfdata:
            v = yfdata.get("trailingEps") or yfdata.get("forwardEps")
            if v is not None:
                try:
                    return float(v), "source=yfinance"
                except Exception:
                    pass
    return None, None

def run(query, topk=5, rebuild_embeddings=False, force_ticker=None):
    df = load_csv()
    print(f"🧾 Loaded {len(df):,} rows from {CSV_PATH}")

    index, emb, meta, df = build_or_load_index(rebuild=rebuild_embeddings)

    I, D = search_index(query, index, k=topk, model_name=EMB_MODEL_NAME)
    retrieved = []
    for idx, score in zip(I, D):
        if idx < 0 or idx >= len(df):
            continue
        text = df.loc[idx, "text_clean"]
        retrieved.append({"idx": int(idx), "score": float(score), "text": text})

    summary = []
    for r in retrieved:
        ent = extract_entities(r["text"])
        summary.append({"idx": r["idx"], "score": r["score"], "text": r["text"], "entities": ent})

    candidate_tickers = []
    candidate_tickers += extract_tickers(query)
    for s in summary:
        candidate_tickers += s["entities"].get("tickers_valid", []) + s["entities"].get("tickers_raw", [])
    candidate_tickers = list(dict.fromkeys([t for t in candidate_tickers if t]))

    chosen_ticker = None
    if force_ticker:
        chosen_ticker = force_ticker.upper()
        print(f"🔁 Forced ticker via CLI: {chosen_ticker}")
    elif candidate_tickers:
        chosen_ticker = candidate_tickers[0]
        print("🔁 Found ticker in text/query ->", chosen_ticker)
    else:
        print("🔁 No tickers found in retrieved chunks or query — attempting company-name -> ticker lookup via Yahoo/AlphaVantage.")
        y_suggestions = yahoo_search_suggestions(query)
        if y_suggestions:
            chosen_ticker = y_suggestions[0]
            print("✅ Yahoo search suggestion ->", chosen_ticker)
        elif ALPHA_KEY:
            av = alpha_symbol_search(query)
            if av:
                chosen_ticker = av[0]
                print("✅ AlphaVantage suggestion ->", chosen_ticker)
        else:
            print("⚠️ No ticker suggestions found (Yahoo possibly rate-limited or AlphaVantage key not set).")

    canonical_eps, eps_meta = canonical_eps_from_results([s["entities"] for s in summary], fallback_ticker=chosen_ticker)
    if canonical_eps is None and eps_meta is None:
        eps_meta = "no_eps_found"

    outname = f"retrieve_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    outpath = Path(EXTRA_OUT_DIR) / outname
    rows = []
    for s in summary:
        ent = s["entities"]
        rows.append({
            "idx": s["idx"],
            "score": s["score"],
            "text": s["text"],
            "eps": json.dumps(ent.get("eps", [])),
            "eps_trend": json.dumps(ent.get("eps_trend", [])),
            "market_cap": json.dumps(ent.get("market_cap", [])),
            "revenue_mention": json.dumps(ent.get("revenue_mention", [])),
            "tickers_raw": json.dumps(ent.get("tickers_raw", [])),
            "tickers_valid": json.dumps(ent.get("tickers_valid", [])),
        })
    pd.DataFrame(rows).to_csv(outpath, index=False, encoding="utf-8")
    print(f"\n💾 Saved extraction results to: {outpath}")

    print("\n--- Extraction summary ---")
    for s in summary:
        ent = s["entities"]
        print(f"\n[{s['idx']}] idx={s['idx']}  score={s['score']:.4f}")
        print(s["text"])
        print("  eps:", ent.get("eps"))
        print("  eps_trend:", ent.get("eps_trend"))
        print("  market_cap:", ent.get("market_cap"))
        print("  revenue:", ent.get("revenue_mention"))
        print("  tickers_raw:", ent.get("tickers_raw"), " tickers_valid:", ent.get("tickers_valid"))

    print("\nCanonical EPS:", canonical_eps, "(", eps_meta, ")")
    print("Chosen ticker (if any):", chosen_ticker)
    return {
        "query": query,
        "chosen_ticker": chosen_ticker,
        "canonical_eps": canonical_eps,
        "eps_meta": eps_meta,
        "summary": summary,
        "output_csv": str(outpath)
    }

def parse_args():
    p = argparse.ArgumentParser(description="Retrieve & extract financial entities (EPS, market cap, revenue, tickers).")
    p.add_argument("--query", required=True, help="Text query to retrieve relevant chunks")
    p.add_argument("--topk", type=int, default=5, help="Number of top chunks to retrieve")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild of embeddings + index")
    p.add_argument("--ticker", type=str, default=None, help="Force ticker (e.g. AAPL) to use for yfinance fallback")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    start = time.time()
    res = run(args.query, topk=args.topk, rebuild_embeddings=args.rebuild, force_ticker=args.ticker)
    print("\nDone. Total time: {:.1f}s".format(time.time() - start))
