import yfinance as yf

# Example company name → ticker mapping
company_tickers = {
    "Apple Inc": "AAPL",
    "Tesla Inc": "TSLA",
    "Pfizer Inc": "PFE",
    "Amazon.com": "AMZN",
    "Meta Platforms": "META"
}

# Lookup function
def get_financial_data(company_name):
    ticker = company_tickers.get(company_name)
    if not ticker:
        return f"❌ No ticker found for {company_name}"

    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "Company": company_name,
        "Ticker": ticker,
        "Market Cap": info.get("marketCap"),
        "Revenue (TTM)": info.get("totalRevenue"),
        "52-Week High": info.get("fiftyTwoWeekHigh"),
        "52-Week Low": info.get("fiftyTwoWeekLow")
    }

# Test lookup
result = get_financial_data("Apple Inc")
print("🔍 Financial Lookup Result:")
print(result)
