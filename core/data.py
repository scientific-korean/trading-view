"""
데이터 수집 → 주봉 변환 → 캐시.

주봉 규약
---------
resample('W-FRI') : 금요일 종료 기준. 금요일 휴장이면 그 주 마지막 거래일 종가.
미완성 주(진행 중인 이번 주)는 반드시 제외한다. 이게 룩어헤드의 흔한 원인.
"""
import os
import datetime as dt
import pandas as pd

CACHE = "cache"


# ------------------------------------------------------------------
# 개별 소스
# ------------------------------------------------------------------
def fetch_yahoo(ticker: str, start: str, retries: int = 3, fresh_within_days: int = 5) -> pd.DataFrame:
    """
    2026-09 ^KS200 사례로 확인된 것: 짧은 구간으로 요청해도, 심지어 방금 직전에
    성공했던 것과 완전히 같은 요청을 다시 보내도 가끔 최근 며칠~몇 주가 빠진
    채로 응답이 온다. 요청 구간 크기 문제가 아니라 야후 쪽의 간헐적(비결정적)
    문제로 보인다 — 재시도하면 대체로 뚫린다. 그래서 결과가 충분히 최신이
    아니면 짧게 쉬었다가 다시 시도하고, 그래도 안 되면 그때까지 받은 것 중
    가장 최신인 걸 반환한다(완전 실패는 아니게).
    """
    import time
    import yfinance as yf
    best = None
    for attempt in range(retries):
        df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)[["open", "high", "low", "close"]]
            # close 가 없는 행은 open/high/low 만 있어도 지표 계산에 그대로 NaN 이
            # 전파된다 (2026-09 ^KS200 에서 실제로 발생: OHL은 있는데 close만 NaN).
            # 전체 결측(dropna(how="all"))보다 엄격하게, close 기준으로 걸러낸다.
            df = df.dropna(subset=["close"])
            if len(df) and (best is None or df.index.max() > best.index.max()):
                best = df
            if len(df) and (pd.Timestamp(dt.date.today()) - df.index.max()).days <= fresh_within_days:
                return df   # 충분히 최신이면 더 재시도할 필요 없음
        if attempt < retries - 1:
            time.sleep(2)
    if best is None or best.empty:
        raise RuntimeError(f"{ticker}: 데이터 없음")
    return best


def fetch_ecos_kr10y(start: str, stat: str, item: str) -> pd.DataFrame:
    """한국은행 ECOS. 환경변수 ECOS_API_KEY 필요."""
    import requests
    key = os.environ.get("ECOS_API_KEY")
    if not key:
        raise RuntimeError("ECOS_API_KEY 미설정")
    s = start.replace("-", "")
    e = dt.date.today().strftime("%Y%m%d")
    url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/100000/"
           f"{stat}/D/{s}/{e}/{item}")
    rows = requests.get(url, timeout=30).json()["StatisticSearch"]["row"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["TIME"], format="%Y%m%d")
    df["close"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    df = df.dropna(subset=["close"]).set_index("date")[["close"]]
    for c in ("open", "high", "low"):
        df[c] = df["close"]
    return df[["open", "high", "low", "close"]]


def fetch_mof_jp10y(start: str) -> pd.DataFrame:
    """일본 재무성 국채금리 CSV (연도별 파일 병합)."""
    import io, requests
    base = "https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv"
    txt = requests.get(base, timeout=60).content.decode("shift_jis", errors="ignore")
    df = pd.read_csv(io.StringIO(txt), skiprows=1)
    df = df.rename(columns={df.columns[0]: "wareki"})

    def to_dt(s):                       # 令和6.4.1 형태 → 서기
        # 실제 CSV는 간지 표기(明治/大正/...)가 아니라 로마자 약자(M/T/S/H/R)를
        # 쓴다 — 예: "S49.9.24", "R8.8.31". 기존 코드는 간지로 매칭해서
        # 단 한 행도 못 걸렀고(전부 NaT), 결과적으로 JP10Y가 늘 빈 데이터였다.
        era = {"M": 1867, "T": 1911, "S": 1925, "H": 1988, "R": 2018}
        for k, base_y in era.items():
            if str(s).startswith(k):
                y, m, d = str(s)[len(k):].split(".")
                return dt.date(base_y + int(y), int(m), int(d))
        return pd.NaT

    df["date"] = pd.to_datetime(df["wareki"].map(to_dt))
    df["close"] = pd.to_numeric(df["10年"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).set_index("date")[["close"]]
    df = df[df.index >= start]
    for c in ("open", "high", "low"):
        df[c] = df["close"]
    return df[["open", "high", "low", "close"]]


# ------------------------------------------------------------------
# 합성 / 주봉
# ------------------------------------------------------------------
def synth_divide(num: pd.DataFrame, den: pd.DataFrame, scale: float = 1.0):
    """원/엔 = USDKRW / USDJPY * 100. 일봉 단계에서 합성해야 정확하다."""
    j = num[["close"]].join(den[["close"]], how="inner", lsuffix="_n", rsuffix="_d")
    out = pd.DataFrame(index=j.index)
    out["close"] = j["close_n"] / j["close_d"] * scale
    for c in ("open", "high", "low"):
        out[c] = out["close"]
    return out[["open", "high", "low", "close"]]


def check_gaps(daily: pd.DataFrame, ticker: str, max_gap_days: int = 15) -> None:
    """
    거래일 기준 비정상적으로 긴 결측 구간을 감지해 경고만 출력한다(중단하지 않음).
    2026-09 ^KS200 에서 약 50일 결측 + 마지막 행 close NaN 이 실제로 발생했는데
    아무 경고 없이 그대로 주봉 지표에 흘러들어갔던 사례가 있어 추가.
    """
    if len(daily) < 2:
        return
    gaps = daily.index.to_series().diff().dt.days
    big = gaps[gaps > max_gap_days]
    for dt_, g in big.items():
        print(f"[경고] {ticker}: {dt_.date()} 직전 {int(g)}일 결측 — 데이터 확인 필요 "
              f"(휴장 며칠 정도는 정상, {max_gap_days}일 초과는 이상 신호)")


def check_freshness(daily: pd.DataFrame, ticker: str, max_age_days: int = 10) -> None:
    """
    check_gaps 는 기존 행들 '사이'의 결측만 잡는다. ^KS200 사례처럼 결측 구간
    뒤에 남는 게 없어 데이터가 그대로 오래된 채 멈춰버리는 경우(가장 최근 행이
    오늘로부터 너무 먼 경우)는 따로 잡아야 한다.
    """
    if len(daily) == 0:
        return
    age = (pd.Timestamp(dt.date.today()) - daily.index.max()).days
    if age > max_age_days:
        print(f"[경고] {ticker}: 최신 데이터가 {daily.index.max().date()}({age}일 전)에 멈춰 있음 "
              f"— 리페치 또는 심볼 확인 필요")


def to_weekly(daily: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
    w = daily.resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna(subset=["close"])

    if drop_incomplete and len(w):
        today = pd.Timestamp(dt.date.today())
        # 마지막 주봉의 라벨(금요일)이 아직 안 지났으면 미완성
        if w.index[-1] >= today:
            w = w.iloc[:-1]
    return w


# ------------------------------------------------------------------
# 진입점
# ------------------------------------------------------------------
def _merge_incremental(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """캐시(과거 전체) + 프레시(최근 재수신분)를 합친다. 겹치는 날짜는 프레시 우선."""
    if fresh.empty:
        return cached
    merged = pd.concat([cached[cached.index < fresh.index.min()], fresh])
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def load(entry: dict, cfg: dict, use_cache: bool = True) -> pd.DataFrame:
    """
    증분 갱신: 캐시가 있으면 21년치를 통째로 재요청하지 않고 캐시 마지막
    날짜보다 14일 전부터만 다시 받아 이어붙인다.

    2026-09 ^KS200 사례로 확인됨 — 같은 티커를 짧은 구간(예: 최근 1개월)으로
    요청하면 최신 데이터가 안정적으로 오는데, 2005년부터 21년치를 통째로
    요청하면 가끔 최근 몇 주가 빠진 채로 응답이 왔다(야후 쪽 문제로 추정,
    결정적이지 않고 간헐적). 요청을 작게 쪼개면 이 문제를 대체로 피해가면서,
    덤으로 매주 21년치를 전부 다시 받을 필요도 없어진다.
    """
    os.makedirs(CACHE, exist_ok=True)
    tkr, start = entry["ticker"], cfg["data"]["start"]
    path = os.path.join(CACHE, f"{tkr.replace('/', '_')}.parquet")
    cached = pd.read_parquet(path) if os.path.exists(path) else None

    if use_cache and cached is not None:
        daily = cached
    else:
        # ecos/mof_jp 는 API 자체가 부분 구간 재수신을 지원하지 않아(항상
        # 전체 이력을 통째로 줌) 증분 갱신 대상에서 제외 — start 그대로 사용.
        incremental = cached is not None and not cached.empty \
            and not entry.get("source") in ("ecos", "mof_jp")
        refetch_from = (cached.index.max() - pd.Timedelta(days=14)).strftime("%Y-%m-%d") \
            if incremental else start

        if "synthetic" in entry:
            s = entry["synthetic"]
            fresh = synth_divide(fetch_yahoo(s["num"], refetch_from),
                                 fetch_yahoo(s["den"], refetch_from),
                                 s.get("scale", 1.0))
        elif entry.get("source") == "ecos":
            fresh = fetch_ecos_kr10y(start, cfg["data"]["ecos_stat_code"],
                                     cfg["data"]["ecos_item_code"])
        elif entry.get("source") == "mof_jp":
            fresh = fetch_mof_jp10y(start)
        else:
            fresh = fetch_yahoo(tkr, refetch_from)

        daily = _merge_incremental(cached, fresh) if incremental else fresh
        daily.to_parquet(path)

    if daily.empty:
        # fetch_yahoo 는 이미 자체적으로 막지만 ecos/mof_jp 경로는 그렇지 않다.
        # 여기서 안 막으면 enrich() 를 거쳐 generate_signal 의 df.iloc[-1] 에서
        # "single positional indexer is out-of-bounds" 라는 알아보기 힘든
        # 에러로 터진다(실제로 JP10Y 에서 이렇게 터졌었음). 원인을 바로 알 수
        # 있게 여기서 막는다.
        raise RuntimeError(f"{tkr}: 수신 데이터 0행 — 소스 응답/파싱 확인 필요")

    check_gaps(daily, tkr)
    check_freshness(daily, tkr)
    return to_weekly(daily)
