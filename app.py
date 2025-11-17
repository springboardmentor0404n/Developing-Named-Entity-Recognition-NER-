# app.py (fixed)
import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import re
import os
import json
import requests
import urllib.parse
from difflib import get_close_matches
from dotenv import load_dotenv
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import math
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY not found in .env — set GEMINI_API_KEY=your_key")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL = "gemini-2.5-flash"  # change if you have different model access
# create a reusable model object
model = genai.GenerativeModel(MODEL)

st.set_page_config(page_title="AI Finance Assistant", layout="wide", page_icon="💼")

# -------------------- HELPERS --------------------
def humanize_value(value):
    try:
        val = float(value)
        if abs(val) >= 1_000_000_000:
            return f"{val/1_000_000_000:.2f} B"
        if abs(val) >= 1_000_000:
            return f"{val/1_000_000:.2f} M"
        if abs(val) >= 1_000:
            return f"{val/1_000:.2f} K"
        return f"{val:,.0f}"
    except:
        return value

def to_number(s):
    if s is None: return None
    if isinstance(s, (int, float)): return float(s)
    s = str(s).strip()
    s2 = s.replace(",", "").replace("$","")
    try:
        if s2.lower().endswith("b"):
            return float(s2[:-1]) * 1_000_000_000
        if s2.lower().endswith("m"):
            return float(s2[:-1]) * 1_000_000
        if s2.lower().endswith("k"):
            return float(s2[:-1]) * 1_000
        return float(re.sub(r"[^\d\.\-]", "", s2))
    except:
        return None

# -------------------- OCR (Gemini Vision then Tesseract fallback) --------------------
def ocr_with_gemini_bytes(file_bytes):
    try:
        resp = model.generate_content([file_bytes, "Extract all readable financial text from this image. Preserve numbers, tables, headings."])
        return resp.text.strip()
    except Exception as e:
        print("Gemini OCR error:", e)
        return ""

def ocr_with_tesseract_bytes(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        text = pytesseract.image_to_string(image)
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        print("Tesseract error:", e)
        return ""

# -------------------- TEXT EXTRACTION --------------------
def extract_text_from_pdf_bytes(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        return " ".join(pages)
    except Exception as e:
        print("PDF text extraction error:", e)
        return ""

def extract_text_from_upload(uploaded_file, prefer_local_ocr=False):
    ft = uploaded_file.type.lower()
    if "pdf" in ft:
        return extract_text_from_pdf_bytes(uploaded_file.read())
    if "text" in ft:
        try:
            return uploaded_file.read().decode("utf-8", errors="ignore")
        except:
            return uploaded_file.read().decode("latin-1", errors="ignore")
    if any(x in ft for x in ("png", "jpg", "jpeg")):
        bytes_data = uploaded_file.read()
        text = ""
        if not prefer_local_ocr:
            text = ocr_with_gemini_bytes(bytes_data)
        if not text or len(text) < 30:
            text = ocr_with_tesseract_bytes(bytes_data)
        return text
    return ""

# -------------------- SEGMENTATION (regex-based) --------------------
def segment_financial_report(text):
    text = text or ""
    sections = {
        "Executive Summary": "",
        "MD&A": "",
        "Financial Statements": "",
        "Financial Statement Subsections": {},
        "Notes": "",
        "Risk Factors": "",
        "Other": ""
    }
    if not text.strip():
        return sections

    patterns = {
        "Executive Summary": r"\b(Executive\s+Summary|Overview)\b",
        "MD&A": r"\b(Management[’']?s\s+Discussion\s+and\s+Analysis|MD&A|Results of Operations|Management Discussion)\b",
        "Financial Statements": r"\b(Consolidated\s+Financial\s+Statements|Financial Statements|Consolidated\s+Statements)\b",
        "Notes": r"\b(Notes to Consolidated Financial Statements|Notes to Financial Statements|Notes)\b",
        "Risk Factors": r"\b(Risk Factors|Market Risk|Legal Proceedings)\b"
    }

    found = {}
    for name, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            found[name] = m.start()

    ordered = sorted(found.items(), key=lambda x: x[1]) if found else []
    for i, (name, start) in enumerate(ordered):
        end = ordered[i+1][1] if i+1 < len(ordered) else len(text)
        chunk = text[start:end].strip()
        if name == "Executive Summary":
            sections["Executive Summary"] = chunk
        elif name == "MD&A":
            sections["MD&A"] = chunk
        elif name == "Financial Statements":
            sections["Financial Statements"] = chunk
        elif name == "Notes":
            sections["Notes"] = chunk
        elif name == "Risk Factors":
            sections["Risk Factors"] = chunk

    # sub-split financial statements
    fs_text = sections["Financial Statements"] or ""
    fs_sub_patterns = {
        "Income Statement": r"Consolidated\s+Statements?\s+of\s+Operations|Statements?\s+of\s+Operations|Income\s+Statement",
        "Comprehensive Income": r"Consolidated\s+Statements?\s+of\s+Comprehensive\s+Income",
        "Balance Sheet": r"Consolidated\s+Balance\s+Sheets|Balance\s+Sheet",
        "Cash Flow": r"Consolidated\s+Statements?\s+of\s+Cash\s+Flows|Statements?\s+of\s+Cash\s+Flows|Cash\s+Flow",
        "Equity Statement": r"Consolidated\s+Statements?\s+of\s+(Stockholders'|Shareholders')\s+Equity"
    }
    starts = []
    for label, pat in fs_sub_patterns.items():
        m = re.search(pat, fs_text, re.IGNORECASE)
        if m:
            starts.append((label, m.start()))
    if starts:
        starts = sorted(starts, key=lambda x: x[1])
        for i, (label, s) in enumerate(starts):
            e = starts[i+1][1] if i+1 < len(starts) else len(fs_text)
            sections["Financial Statement Subsections"][label] = fs_text[s:e].strip()

    if all(not sections[k].strip() for k in ["Executive Summary","MD&A","Financial Statements","Notes","Risk Factors"]):
        sections["Other"] = text[:5000]

    if re.search(r"\b(chart|graph|figure)\b", text, re.IGNORECASE):
        sections["Charts / Graphs"] = "Charts/graphs likely present (images extracted if PDF)."

    return sections

# -------------------- PDF TABLES & IMAGE EXTRACTION --------------------
def extract_tables_from_pdf_path(pdf_path, max_pages=50):
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = min(len(pdf.pages), max_pages)
            for i in range(pages):
                page = pdf.pages[i]
                tables = page.extract_tables()
                for t in tables:
                    df = pd.DataFrame(t)
                    results.append({"page": i+1, "data": df})
    except Exception as e:
        print("pdfplumber error:", e)
    return results

def extract_images_from_pdf_path(pdf_path, out_dir="extracted_images"):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    try:
        doc = fitz.open(pdf_path)
        for pageno in range(len(doc)):
            imgs = doc.get_page_images(pageno)
            for idx, img in enumerate(imgs, start=1):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                out = os.path.join(out_dir, f"page{pageno+1}_img{idx}.png")
                if pix.n < 5:
                    pix.save(out)
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(out)
                saved.append(out)
    except Exception as e:
        print("image extraction error:", e)
    return saved

# -------------------- TICKER LOOKUP --------------------
def find_ticker_for_company(company):
    if not company:
        return None
    query = re.sub(r",? Inc\.?|,? Ltd\.?|,? Corp\.?|,? Corporation", "", company, flags=re.I).strip()
    encoded = urllib.parse.quote_plus(query)
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={encoded}"
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            result = resp.json()
            candidates = []
            for q in result.get("quotes", []):
                sym = q.get("symbol")
                name = q.get("shortname") or q.get("longname") or ""
                if sym:
                    candidates.append((sym, name))
            if candidates:
                names = [n for _, n in candidates if n]
                if names:
                    best = get_close_matches(query, names, n=1, cutoff=0.6)
                    if best:
                        for sym, name in candidates:
                            if name == best[0]:
                                return sym
                return candidates[0][0]
    except Exception as e:
        print("yahoo search error:", e)
    fallback = {"netflix":"NFLX","google":"GOOG","apple":"AAPL","microsoft":"MSFT","amazon":"AMZN"}
    qlow = query.lower()
    for k,v in fallback.items():
        if k in qlow:
            return v
    return None

# -------------------- RULE-BASED ENTITY PARSING --------------------
def parse_simple_entities(text):
    out = {}
    m = re.search(r"(?:Company Name|Registrant|THE COMPANY|Company)\s*[:\-\n]?\s*([A-Za-z0-9&\.,\s\(\)'\-]{3,200})", text)
    if not m:
        m = re.search(r"\n([A-Z][A-Za-z0-9&\.,\s]{3,120}?(?:,?\sInc\.|,?\sCorporation|,?\sLLC|,?\sLtd\.))", text)
    out["Company Name"] = m.group(1).strip() if m else "N/A"
    m = re.search(r"(?:For the (?:year|quarter) ended|Period Ended|As of)\s+([A-Za-z0-9,\s]+?\d{4})", text, re.IGNORECASE)
    out["Report Period"] = m.group(1).strip() if m else "N/A"

    patterns = {
        "Revenue": r"Revenue[s]?\s*[^\d\n\r]{0,6}([\d,]+(?:\.\d+)?)",
        "Net Income": r"Net\s+Income\s*[^\d\n\r]{0,6}([\d,]+(?:\.\d+)?)",
        "Operating Income": r"Operating\s+Income\s*[^\d\n\r]{0,6}([\d,]+(?:\.\d+)?)",
        "EPS": r"(?:Earnings Per Share|EPS)[^\d\n\r]{0,6}([\d\.]+)",
        "Total Assets": r"Total\s+Assets\s*[^\d\n\r]{0,6}([\d,]+(?:\.\d+)?)",
        "Total Liabilities": r"Total\s+Liabilities\s*[^\d\n\r]{0,6}([\d,]+(?:\.\d+)?)",
        "Shareholder Equity": r"(?:Total\s+Stockholders'?|Total\s+Shareholders'?)\s+Equity\s*[^\d\n\r]{0,6}([\d,]+(?:\.\d+)?)",
        "Cash Flow": r"(?:Net\s+Cash\s+Provided|Net\s+cash\s+provided|Net Cash Flow|Cash Flow)[^\d\n\r]{0,6}([\d,]+(?:\.\d+)?)",
        "Currency": r"\b(USD|US\$|EUR|INR|GBP|JPY)\b"
    }
    for k,p in patterns.items():
        m = re.search(p, text, re.IGNORECASE)
        out[k] = m.group(1).strip() if m else "N/A"
    for k in list(out.keys()):
        if k not in ("Company Name", "Report Period", "Currency") and out[k] != "N/A":
            try:
                num = float(out[k].replace(",", ""))
                out[k] = humanize_value(num)
            except:
                pass
    return out

# -------------------- STREAMLIT UI --------------------
st.title("AI Finance Assistant")

uploaded_file = st.file_uploader("Upload file", type=["pdf","txt","png","jpg","jpeg"])

if uploaded_file:
    st.info("Reading file and extracting text...")
    # Save for pdf tools if pdf
    is_pdf = "pdf" in uploaded_file.type.lower()
    temp_pdf = None
    if is_pdf:
        temp_pdf = "temp_report.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Extract text (images: Gemini then Tesseract fallback)
    text = extract_text_from_upload(uploaded_file, prefer_local_ocr=False)
    if not text or len(text.strip()) < 20:
        st.warning("Extracted text appears short — retrying with local OCR fallback.")
        # rewind for reading again (pdf handled already by saved file)
        if is_pdf:
            with open(temp_pdf, "rb") as f:
                text = extract_text_from_pdf_bytes(f.read())
        else:
            try:
                uploaded_file.seek(0)
            except:
                pass
            text = extract_text_from_upload(uploaded_file, prefer_local_ocr=True)

    if not text or len(text.strip()) < 10:
        st.error("Could not extract readable text from this file.")
    else:
        st.success("Text extracted — running summary and extraction...")

        # ---------- SUMMARY (Gemini) ----------
        st.header("1) Document Summary")
        try:
            # Use a chunk of text for summary (Gemini has token limits)
            chunk = text[:11000]
            prompt = f"Summarize the following financial document concisely (3-6 bullet points). Text: {chunk}"
            resp = model.generate_content(prompt)
            summary = resp.text.strip()
            st.markdown(summary if summary else "No summary returned.")
        except Exception as e:
            st.error(f"Summary (Gemini) failed: {e}")
            # fallback: show first 800 chars
            st.write(text[:800] + ("..." if len(text)>800 else ""))
            summary = text[:800]

        # ---------- ENTITY EXTRACTION (Gemini primary, local fallback) ----------
        st.header("2) Extracted Financial Entities")
        extracted_entities = {}
        try:
            prompt_ent = f"""
            Extract key financial entities from the text and return a JSON object with fields:
            Company Name, Report Period, Revenue, Net Income, Operating Income, EPS, Total Assets, Total Liabilities, Shareholder Equity, Cash Flow, Currency, Key Risks (list).
            If a field is missing, set it to "N/A".
            Text: {text[:12000]}
            """
            resp_e = model.generate_content(prompt_ent)
            out_e = resp_e.text.strip()
            json_match = re.search(r"\{[\s\S]*\}", out_e)
            extracted_entities = json.loads(json_match.group()) if json_match else {}
        except Exception as e:
            print("Gemini entity error:", e)
            # fallback to rule-based parser
            extracted_entities = parse_simple_entities(text)

        # fill missing keys from rule-based fallback
        fallback = parse_simple_entities(text)
        keys_needed = ["Company Name","Report Period","Revenue","Net Income","Operating Income","EPS","Total Assets","Total Liabilities","Shareholder Equity","Cash Flow","Currency","Key Risks"]
        for k in keys_needed:
            if k not in extracted_entities or not extracted_entities.get(k):
                extracted_entities[k] = fallback.get(k, "N/A")

        # Display entities nicely
        for k,v in extracted_entities.items():
            st.markdown(f"**{k}:** {v}")

        # ---------- SEGMENTATION (rule-based only) ----------
        st.header("3) Document Segmentation")
        segments = segment_financial_report(text)
        st.write("Detected sections (expand to read). Use 'Refine segmentation with Gemini' only if you want summaries for each section (uses API).")
        for sec_name, sec_content in segments.items():
            with st.expander(f"{sec_name}", expanded=False):
                if isinstance(sec_content, dict):
                    for sub, subtext in sec_content.items():
                        st.markdown(f"**{sub}**")
                        st.write(subtext[:3000] + ("..." if len(subtext)>3000 else ""))
                else:
                    st.write(sec_content[:3000] + ("..." if len(sec_content)>3000 else ""))

        if st.button("Refine segmentation with Gemini (summarize each detected section)"):
            with st.spinner("Refining segments..."):
                refined = {}
                for name, content in segments.items():
                    if isinstance(content, dict):
                        refined[name] = {}
                        for sub, subtext in content.items():
                            prompt_s = f"Shortly summarize this section (2-4 bullets): {subtext[:9000]}"
                            try:
                                r = model.generate_content(prompt_s)
                                refined[name][sub] = r.text.strip()
                            except Exception as e:
                                refined[name][sub] = subtext[:800]
                    else:
                        prompt_s = f"Shortly summarize this section (2-4 bullets): {content[:9000]}"
                        try:
                            r = model.generate_content(prompt_s)
                            refined[name] = r.text.strip()
                        except Exception as e:
                            refined[name] = content[:800]
                st.json(refined)

        # ---------- TABLE EXTRACTION (pdfplumber) ----------
        st.header("4) Table Extraction (Streamlit view)")
        if is_pdf and os.path.exists(temp_pdf):
            tables = extract_tables_from_pdf_path(temp_pdf, max_pages=50)
            if tables:
                st.write(f"Found {len(tables)} table(s) (showing up to first 10).")
                for t in tables[:10]:
                    st.subheader(f"Page {t['page']}")
                    df = t["data"].copy()
                    st.dataframe(df.replace({None:""}))
            else:
                st.info("No structured tables detected (via pdfplumber).")
        else:
            st.info("Table extraction only available for PDF uploads.")

        # ---------- IMAGE / CHART EXTRACTION ----------
        

        # ---------- YAHOO FINANCE VALIDATION ----------
        st.header("6) Yahoo Finance Validation")
        company_guess = extracted_entities.get("Company Name") or extracted_entities.get("company_name") or ""
        st.write(f"Guessed Company: **{company_guess}**")
        manual = st.text_input("If auto lookup fails, enter ticker (e.g., NFLX):", value="")
        ticker = manual.strip().upper() if manual.strip() else find_ticker_for_company(company_guess)
        yahoo_data = {}
        if ticker:
            try:
                st.write(f"Using ticker: **{ticker}**")
                stock = yf.Ticker(ticker)
                info = stock.info
                yahoo_data = info
                st.write(f"**{info.get('longName', company_guess)} ({ticker})**")
                st.write(f"Current Price: {info.get('currentPrice', 'N/A')}")
                st.write(f"Market Cap: {humanize_value(info.get('marketCap', 'N/A'))}")
                st.write(f"PE Ratio: {info.get('trailingPE', 'N/A')}")
                st.write(f"52W Range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.markdown(f"[View on Yahoo Finance](https://finance.yahoo.com/quote/{ticker})")
                st.session_state["yahoo_data"] = yahoo_data
            except Exception as e:
                st.warning(f"Yahoo fetch failed: {e}")
        else:
            st.info("Ticker not found automatically. Provide manual ticker if desired.")

        # ---------- Q&A CHATBOT ----------
        st.markdown("---")
        st.markdown("## 💬 Ask Anything About This Report")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # use streamlit text_input for simplicity and speed
        user_query = st.text_input("Ask a question (e.g., What is the total revenue growth?)")
        if user_query:
            with st.spinner("Thinking..."):
                context = f"""
The user uploaded a financial report.
Extracted entities: {json.dumps(extracted_entities, indent=2)}.
Yahoo Finance verification: {json.dumps(st.session_state.get('yahoo_data', {}), indent=2)}.
Full extracted text (excerpt): {text[:8000]}.
Answer the following question clearly and accurately:
{user_query}
"""
                try:
                    resp = model.generate_content(context)
                    answer = resp.text.strip()
                except Exception as e:
                    answer = f"Error generating answer: {e}"

            st.session_state["chat_history"].append({"user": user_query, "bot": answer})

        for chat in st.session_state["chat_history"]:
            st.markdown(f"<div class='chat-box'><div class='user-msg'>User:</div> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-box'><div class='bot-msg'>AI:</div> {chat['bot']}</div>", unsafe_allow_html=True)

        # ---------- DOWNLOAD ENTITIES ----------
        st.header("Download")
        st.download_button("Download extracted entities (JSON)", json.dumps(extracted_entities, indent=2), file_name="entities.json", mime="application/json")

    # cleanup
    if is_pdf and temp_pdf and os.path.exists(temp_pdf):
        try:
            os.remove(temp_pdf)
        except:
            pass
else:
    st.info("Upload a PDF, TXT, PNG, or JPG to begin.")
