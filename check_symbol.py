import yfinance as yf

print("=== ^KS200 ===")
print(yf.download("^KS200", period="1mo").tail())

print("\n=== KOSPI200.KS ===")
print(yf.download("KOSPI200.KS", period="1mo").tail())
