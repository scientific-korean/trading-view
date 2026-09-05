"""
^KS200: period="1mo"로는 오늘(9/4) 데이터가 정상 나오는데, run_weekly.py 처럼
start="2005-01-01"로 긴 구간을 요청하면 여전히 2026-07-16에서 멈춘다.
요청 구간 길이에 따라 어디서부터 잘리는지 좁혀서 찾아본다.
"""
import yfinance as yf

for start in ["2026-08-01", "2026-07-01", "2026-01-01", "2020-01-01", "2005-01-01"]:
    df = yf.download("^KS200", start=start, progress=False, auto_adjust=False)
    last = df.index.max() if len(df) else None
    print(f"start={start:<12} rows={len(df):<6} last_date={last}")
