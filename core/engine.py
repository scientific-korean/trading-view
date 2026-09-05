"""
신호 순차 실행. 백테스트와 캘리브레이션이 공유한다.

룩어헤드 차단: generate_signal 에 df.iloc[:i+1] 만 넘긴다.
이 슬라이싱이 구조적 안전장치이므로 절대 바꾸지 말 것.
"""
import pandas as pd
from .signals import generate_signal


def run_signals(df: pd.DataFrame, params: dict, meta: dict,
                warmup: int = None, return_state: bool = False):
    """return_state=True 면 (DataFrame, 최종 state) 를 반환."""
    if warmup is None:
        warmup = params["slow"] + params["signal"]

    state, recs = {}, []
    for i in range(warmup, len(df)):
        window = df.iloc[: i + 1]          # ← 미래 차단
        dec, state = generate_signal(window, state, params, meta)
        rec = {k: v for k, v in dec.items() if k != "flags"}
        rec["date"] = df.index[i]
        rec["close"] = float(df["close"].iloc[i])
        rec["vol_warning"] = dec["flags"]["vol_warning"]
        recs.append(rec)

    out = pd.DataFrame(recs).set_index("date") if recs else pd.DataFrame()
    return (out, state) if return_state else out
