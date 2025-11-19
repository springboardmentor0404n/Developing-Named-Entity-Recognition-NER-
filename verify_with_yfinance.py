# verify_with_yfinance.py
import yfinance as yf

def verify_ticker(ticker):
    obj = yf.Ticker(ticker)
    info = obj.info
    return {
        "symbol": ticker,
        "shortName": info.get("shortName"),
        "marketCap": info.get("marketCap"),
        "previousClose": info.get("previousClose"),
        "regularMarketPrice": info.get("regularMarketPrice")
    }

if __name__ == "__main__":
    print(verify_ticker("AAPL"))
