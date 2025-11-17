import streamlit as st
import google.generativeai as genai
import fitz
import re
import os
import json
import yfinance as yf
import requests
from dotenv import load_dotenv

# -------------------- LOAD GEMINI API --------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ Gemini API key not found in .env file.")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Finance", layout="centered")

st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #E0E0E0; font-family: "Segoe UI", sans-serif; }
h1 { color: #57A6FF; text-align: center; }
hr { border: 1px solid #2D2F39; margin: 20px 0; }
.chat-box { background-color: #1E1E2E; padding: 15px; border-radius: 10px; margin-top: 10px; }
.user-msg { color: #57A6FF; font-weight: bold; }
.bot-msg { color: #E0E0E0; }
</style>
""", unsafe_allow_html=True)

st.title("AI Finance Assistant")

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("📎 Upload Financial Report (PDF or TXT)", type=["pdf", "txt"])

def extract_text(file):
    """Extract text from PDF or TXT"""
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = " ".join([page.get_text("text") for page in doc])
    else:
        text = file.read().decode("utf-8", errors="ignore")
    return re.sub(r"\s+", " ", text).strip()

# -------------------- UTILITY FUNCTIONS --------------------
def humanize_value(value):
    """Convert numeric values to human-readable format"""
    try:
        val = float(value)
        if abs(val) >= 1_000_000_000:
            return f"{val/1_000_000_000:.2f} B"
        elif abs(val) >= 1_000_000:
            return f"{val/1_000_000:.2f} M"
        elif abs(val) >= 1_000:
            return f"{val/1_000:.2f} K"
        else:
            return f"{val:,.0f}"
    except:
        return value

# -------------------- ENTITY EXTRACTION --------------------
if uploaded_file:
    st.info("📖 Reading and extracting entities...")
    text_input = extract_text(uploaded_file)

    with st.spinner("🧠 Extracting key financial entities..."):
        prompt = f"""
        Extract key financial entities from the following text and return them as a JSON.
        Required fields:
        - Company Name
        - Report Period
        - Revenue
        - Net Income
        - Operating Income
        - Earnings Per Share (EPS)
        - Total Assets
        - Total Liabilities
        - Shareholder Equity
        - Cash Flow
        - Currency
        If any field not found, use "N/A".
        Text: {text_input[:12000]}
        """
        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt)
        output = response.text.strip()

    st.success("✅ Entities extracted successfully!")

    try:
        json_match = re.search(r"\{[\s\S]*\}", output)
        data = json.loads(json_match.group()) if json_match else {}
    except:
        data = {"Error": "Could not parse JSON", "Raw Output": output}

    # Humanize financial numbers
    for key, val in data.items():
        if any(term in key.lower() for term in ["revenue", "income", "assets", "liabilities", "equity", "cash"]):
            data[key] = humanize_value(val)

    st.markdown("### 📌 Extracted Financial Entities")
    for k, v in data.items():
        st.markdown(f"**{k}:** {v}")

    st.session_state["extracted_entities"] = data
    st.session_state["extracted_text"] = text_input

    # -------------------- YAHOO FINANCE --------------------
    company = data.get("Company Name", "")
    if company and company != "N/A":
        st.markdown("---")
        st.markdown("### 📊 Yahoo Finance Verification")

        try:
            query = re.sub(
                r",? Inc\.?|,? Ltd\.?|,? Corp\.?|,? Corporation",
                "",
                company,
                flags=re.I,
            ).strip()

            query_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
            response = requests.get(query_url, timeout=10)

            if response.status_code == 200 and response.text.strip():
                result_json = response.json()
                results = result_json.get("quotes", [])
                ticker = results[0]["symbol"] if results else None
            else:
                ticker = None

            # fallback mapping
            if not ticker:
                mapping = {
                    "netflix": "NFLX",
                    "google": "GOOG",
                    "apple": "AAPL",
                    "microsoft": "MSFT",
                    "amazon": "AMZN",
                }
                for key, val in mapping.items():
                    if key in query.lower():
                        ticker = val

            if ticker:
                stock = yf.Ticker(ticker)
                info = stock.info

                st.write(f"**Company:** {info.get('longName', company)} ({ticker})")
                st.write(f"**Current Price:** ${info.get('currentPrice', 'N/A')}")
                st.write(f"**Market Cap:** {humanize_value(info.get('marketCap', 'N/A'))}")
                st.write(f"**PE Ratio:** {info.get('trailingPE', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**52 Week Range:** {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}")

                yahoo_link = f"https://finance.yahoo.com/quote/{ticker}"
                st.markdown(f"🔗 **Verify on Yahoo Finance:** [{ticker}]({yahoo_link})")

                st.session_state["yahoo_data"] = info

            else:
                st.warning(f" Could not find ticker for {company} on Yahoo Finance.")

        except Exception as e:
            st.error(f" Yahoo Finance fetch failed: {e}")

    # -------------------- Q&A SECTION --------------------
    st.markdown("---")
    st.markdown("## 💬 Ask Anything About This Report")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_query = st.chat_input("Ask a question (e.g., What is the total revenue growth?)")

    if user_query:
        with st.spinner("Thinking..."):
            context = f"""
            The user uploaded a financial report.
            Extracted data: {json.dumps(data, indent=2)}.
            Yahoo Finance verification: {json.dumps(st.session_state.get('yahoo_data', {}), indent=2)}.
            Full extracted text: {text_input[:8000]}.
            Answer the following question clearly and accurately:
            {user_query}
            """
            response = model.generate_content(context)
            answer = response.text.strip()

        st.session_state["chat_history"].append({"user": user_query, "bot": answer})

    for chat in st.session_state["chat_history"]:
        st.markdown(
            f"<div class='chat-box'><div class='user-msg'>User:</div> {chat['user']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='chat-box'><div class='bot-msg'>AI:</div> {chat['bot']}</div>",
            unsafe_allow_html=True,
        )

#python -m streamlit run app.py 