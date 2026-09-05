"""
지표 계산.

핵심 규약
---------
kind='price' : PPO 사용. (EMA_fast - EMA_slow) / EMA_slow * 100  → 단위 %
kind='rate'  : MACD 원값 사용. EMA_fast - EMA_slow               → 단위 %p
               (금리는 이미 % 단위라 정규화 불필요하고,
                0 근처/마이너스에서 PPO 분모가 폭발하므로 금지)

RSI 는 Wilder 평활(alpha = 1/period). 국내 HTS 표준과 일치.
단순이동평균 방식으로 짜면 증권사 앱 값과 어긋나므로 주의.
"""
import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _wilder_smooth(x: pd.Series, period: int) -> pd.Series:
    """
    정통 Wilder 평활.
      시드   : 최초 period 개의 단순평균
      이후   : avg[t] = (avg[t-1]*(period-1) + x[t]) / period

    주의 - ewm(alpha=1/period) 만 쓰면 시드가 첫 값 하나가 되어
    증권사 앱 값과 어긋난다. 시드를 단순평균으로 잡아야 일치한다.
    """
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    valid = np.where(~np.isnan(v))[0]
    if len(valid) < period:
        return pd.Series(out, index=x.index)

    s = valid[period - 1]
    acc = np.nanmean(v[valid[0]: s + 1])
    out[s] = acc
    for i in range(s + 1, len(v)):
        if np.isnan(v[i]):
            out[i] = acc
            continue
        acc = (acc * (period - 1) + v[i]) / period
        out[i] = acc
    return pd.Series(out, index=x.index)


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. 국내 HTS 표준과 일치."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)          # 하락 전무 → 100
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)  # 상승 전무 → 0
    return rsi


def macd_ppo(close: pd.Series, fast: int, slow: int, signal: int,
             kind: str = "price") -> pd.DataFrame:
    """
    kind='price' → PPO (%)
    kind='rate'  → MACD 원값 (%p)
    반환 컬럼: osc_line, osc_signal, osc_hist
    """
    ef, es = _ema(close, fast), _ema(close, slow)

    if kind == "rate":
        line = ef - es
    else:
        # 분모 보호: EMA_slow 가 0 이하로 갈 수 있는 자산은 rate 로 분류해야 함
        denom = es.replace(0.0, np.nan)
        line = (ef - es) / denom.abs() * 100.0

    sig = _ema(line, signal)
    return pd.DataFrame({
        "osc_line": line,
        "osc_signal": sig,
        "osc_hist": line - sig,
    }, index=close.index)


def bollinger(close: pd.Series, period: int, n_std: float,
              kind: str = "price") -> pd.DataFrame:
    """
    bb_width:
      price → (상단-하단) / 중심선          (비율)
      rate  → (상단-하단) * 100             (bp)
              금리는 중심선이 0 근처면 비율이 발산하므로 절대폭 사용
    """
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper, lower = mid + n_std * sd, mid - n_std * sd

    if kind == "rate":
        width = (upper - lower) * 100.0
    else:
        width = (upper - lower) / mid.replace(0.0, np.nan)

    pctb = (close - lower) / (upper - lower).replace(0.0, np.nan)

    return pd.DataFrame({
        "bb_upper": upper, "bb_mid": mid, "bb_lower": lower,
        "bb_width": width, "bb_pctb": pctb,
    }, index=close.index)


def bb_width_percentile(width: pd.Series, window: int) -> pd.Series:
    """
    최근 window(주) 구간 내에서 현재 BB width가 상위 몇 %에 해당하는지(0~100).

    절대 임계치(예: 0.8) 하나로는 자산별 형평이 안 맞는다 — 자산마다 BB width의
    타고난 스케일이 크게 다르다(예: 2026-09 기준 브렌트유 0.47 vs 원/달러 0.15,
    실제로 현재 절대 임계치 0.8은 대부분 자산에서 사실상 발동하지 않는다는 README의
    기존 지적). 그래서 절대값 대신 "그 자산 자신의 최근 분포 안에서 지금이 상대적으로
    넓은 편인가"로 판정한다. price/rate 스케일이 달라도 그대로 통하므로 kind별로
    나눌 필요도 없어진다.

    전체 히스토리(확장창) 대신 롤링 윈도우를 쓰는 이유는 변동성 레짐이 수년 단위로
    바뀌기 때문 — 옛날 저변동기/고변동기 기록이 계속 기준에 섞이면 최근 레짐 기준
    "지금이 넓다"는 판단이 무뎌진다.

    window 만큼 데이터가 쌓이기 전에는 NaN(경고 판정 보류 — 관망과 동일한 취급).
    """
    def pct_rank(x):
        return (x <= x[-1]).mean() * 100.0
    return width.rolling(window, min_periods=window).apply(pct_rank, raw=True)


def slope(series: pd.Series, window: int = 3) -> pd.Series:
    """
    최근 window 봉 평균 기울기 = (x[t] - x[t-window]) / window
    주봉 1봉 차분은 노이즈가 커서 평균 차분을 사용.
    """
    return (series - series.shift(window)) / float(window)


def enrich(df: pd.DataFrame, p: dict, kind: str = "price") -> pd.DataFrame:
    """주봉 OHLC 에 전체 지표를 붙인다."""
    out = df.copy()
    c = out["close"]

    out = out.join(macd_ppo(c, p["fast"], p["slow"], p["signal"], kind))
    out["rsi"] = rsi_wilder(c, p["rsi_period"])
    out = out.join(bollinger(c, p["bb_period"], p["bb_std"], kind))
    out["bb_width_pctile"] = bb_width_percentile(out["bb_width"], p["bb_width_window_weeks"])
    out["osc_slope"] = slope(out["osc_line"], p["slope_window"])
    return out
