# app.py (modified for Milestone 3→4)
import streamlit as st
from datetime import datetime, date
from typing import List, Tuple, Optional
import io, json

# --- import your backend functions ---
from finance_insight_backend import analyze_text, analyze_pdf_file

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Page config & basic styles
st.set_page_config(page_title="Financial Insight | Chat With Your Document", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background-color: #f6f8fa;}
    .report-card {background: #fff; border-radius: 14px; box-shadow: 0 2px 12px #eee; padding: 18px; margin-bottom: 18px;}
    .chat-bubble {background: #e8eef6; border-radius: 18px; padding: 10px 12px; margin: 12px 0;}
    .chat-bubble.user {background: #cbe9de;}
    .chat-entity {background: linear-gradient(90deg,#e6ffc0,#eeebff); border-radius: 8px; padding: 4px 12px; display: inline-block; margin-right: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 Financial Insight | Chat With Your Document")

# Sidebar: settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bank-building.png", width=64)
    st.markdown("## Settings")
    entities: List[str] = st.multiselect(
        "Entities to extract",
        ["market_cap", "EPS", "revenue_growth", "stock_price_trend", "dividend_yield", "pe_ratio"],
        default=["market_cap", "EPS", "revenue_growth"],
    )
    events: List[str] = st.multiselect(
        "Events to detect", ["IPO", "M&A", "earnings_call", "dividend"], default=["IPO", "M&A", "earnings_call"]
    )
    conf = st.slider("Confidence threshold", 0.0, 0.99, 0.50, 0.01)
    use_time = st.checkbox("Filter events by date", value=False)
    start_dt, end_dt = None, None
    if use_time:
        s = st.date_input("Start", value=date(2022, 1, 1))
        e = st.date_input("End", value=date.today())
        start_dt = datetime.combine(s, datetime.min.time())
        end_dt = datetime.combine(e, datetime.min.time())

# Upload area
st.markdown('<div class="report-card">', unsafe_allow_html=True)
st.subheader("🗂️ Upload Your Financial Report")

uploaded = st.file_uploader("Upload (.pdf, .txt) or paste text below", type=["pdf", "txt"])
raw_text = st.text_area("Paste text here", height=160, placeholder="Paste financial text, a news snippet, or report details…")
st.markdown('</div>', unsafe_allow_html=True)

def _read_uploaded_text(file) -> str:
    if file is None:
        return ""
    if file.type == "text/plain":
        return file.read().decode("utf-8", errors="ignore")
    if file.type == "application/pdf" and PyPDF2 is not None:
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""

analyze_btn = st.button("💡 Analyze", use_container_width=True)

if analyze_btn:
    # determine input source
    if not raw_text.strip() and uploaded is None:
        st.warning("⚠️ Please upload a PDF / TXT or paste text in the box.")
    else:
        with st.spinner("🔎 Analyzing your document…"):
            timeframe: Optional[Tuple[Optional[datetime], Optional[datetime]]] = (start_dt, end_dt) if use_time else None

            result = {}
            # If PDF uploaded -> use pdf pipeline (better table + segmentation)
            if uploaded is not None and uploaded.type == "application/pdf":
                # we must pass a file-like object accepted by pdfplumber
                uploaded_bytes = uploaded.read()
                file_like = io.BytesIO(uploaded_bytes)
                try:
                    result = analyze_pdf_file(file_like)
                    # merge in entity/event extraction too (use analyze_text on cleaned text)
                    text_for_entities = result.get("text", "")
                    ent_evt = analyze_text(text_for_entities, entities, events, conf_threshold=conf, timeframe=timeframe)
                    # put entities/events/verified under consistent keys
                    result["entities"] = ent_evt.get("entities", {})
                    result["events"] = ent_evt.get("events", {})
                    # merged verified already from pdf pipeline may exist; prefer ent_evt verified if present
                    if ent_evt.get("verified"):
                        result["verified"] = ent_evt.get("verified")
                except Exception as e:
                    # fallback to text extraction + analyze_text
                    txt = _read_uploaded_text(uploaded)
                    result = analyze_text(txt, entities, events, conf_threshold=conf, timeframe=timeframe)
                    result["error"] = f"PDF pipeline error, fell back to text path: {str(e)}"
            else:
                # text path (pasted or .txt upload)
                text_input = raw_text.strip() or _read_uploaded_text(uploaded)
                result = analyze_text(text_input, entities, events, conf_threshold=conf, timeframe=timeframe)

        # --- Display results ---
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown('<div class="chat-bubble">✅ <b>Successfully loaded your report.</b></div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-bubble user">💬 <b>Question:</b> What does this report contain?</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble">🧾 <b>Summary:</b> {result.get("summary", "AI summary coming soon…")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Sections (segmentation)
        if result.get("sections"):
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("📚 Document Segmentation")
            for name, snippet in result["sections"].items():
                if not snippet:
                    st.markdown(f"**{name}** — _not found_")
                else:
                    with st.expander(f"{name}", expanded=False):
                        st.write(snippet[:2000])
            st.markdown('</div>', unsafe_allow_html=True)

        # Entities
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("🔎 Extracted Entities")
        ents = result.get("entities", {})
        if not ents:
            st.caption("— none —")
        else:
            for name, items in ents.items():
                st.markdown(f"<span class='chat-entity'>{name}</span>", unsafe_allow_html=True)
                if not items:
                    st.caption("— none —")
                else:
                    for it in items[:10]:
                        label = it.get("raw") or it.get("text") or str(it)
                        st.write("•", label)
        st.markdown('</div>', unsafe_allow_html=True)

        # Events
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("📢 Detected Events")
        evs = result.get("events", {})
        if not evs or not any(evs.values()):
            st.info("No financial events detected.")
        else:
            for name, items in evs.items():
                st.markdown(f"<span class='chat-entity'>{name}</span>", unsafe_allow_html=True)
                if not items:
                    st.caption("— none —")
                else:
                    st.json(items, expanded=False)
        st.markdown('</div>', unsafe_allow_html=True)

        # Verified tickers
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("✅ Verified (Yahoo Finance)")
        ver = result.get("verified", {})
        ticks = ver.get("tickers", []) if isinstance(ver, dict) else []
        if ticks:
            st.table(ticks)
        else:
            st.caption("No tickers verified in this text.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Tables (if any) — show up to first 6 parsed tables
        parsed_tables = result.get("tables") or []
        if parsed_tables:
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("📊 Parsed Tables (first 6)")
            show_n = min(len(parsed_tables), 6)
            for i, t in enumerate(parsed_tables[:show_n], start=1):
                st.markdown(f"**Table {i} — page {t.get('page')} — type: {t.get('type', 'Unknown')}**")
                raw_df = t.get("raw")
                numeric_df = t.get("numeric")
                if raw_df is not None:
                    st.markdown("Raw table preview (first 6 rows):")
                    try:
                        st.dataframe(raw_df.head(6))
                    except Exception:
                        st.write(raw_df.head(6).to_string())
                if numeric_df is not None:
                    st.markdown("Normalized numeric preview (first 6 rows):")
                    try:
                        st.dataframe(numeric_df.head(6))
                    except Exception:
                        st.write(numeric_df.head(6).to_string())
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)

        # Raw text viewer (collapsible)
        if result.get("text"):
            with st.expander("🔍 View cleaned document text (first 20k chars)"):
                st.text(result.get("text")[:20000])

        # Error (if any)
        if result.get("error"):
            st.error(result.get("error"))

        # Download JSON of extracted result
        btn_col1, btn_col2 = st.columns([1, 3])
        with btn_col1:
            js = json.dumps(result, default=str, indent=2)
            st.download_button("⬇️ Download JSON", data=js, file_name="finance_insight_result.json", mime="application/json")
        with btn_col2:
            st.success("Analysis complete ✅")
