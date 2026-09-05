"""
하이퍼파라미터 후보들을 바꿔가며:
  1) 회전 지연(turn lag) — 매수확정이 실제 저점보다 몇 주 늦게/매도확정이 실제
     고점보다 몇 주 늦게 잡히는지
  2) 기준선 대비 초과정확도 (기존 decompose.py 방식)
  3) 연간 확정전환 횟수 (신호 빈도 = 휩쏘 위험 프록시)
를 같은 캐시 데이터로 재계산해서 비교한다. 라이브 config.yaml은 건드리지 않는다.

(2026-09: 코스피 지수 티커를 ^KS200→^KS11로 교체함 — 예전 ^KS200 결측
문제로 EXCLUDE 처리했던 건 이제 무의미해져서 제거함.)
"""
import sys, os, json
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import data as D
from core.indicators import enrich
from core.engine import run_signals

HORIZONS = [1, 4, 13]
EXCLUDE = set()

CANDIDATES = {
    "A 현재(baseline)":         dict(fast=19, slow=39, signal=14, rsi_period=14, rsi_upper=55, rsi_lower=45),
    "B signal만 단축(14→7)":     dict(fast=19, slow=39, signal=7,  rsi_period=14, rsi_upper=55, rsi_lower=45),
    "C EMA+signal 단축(12/26/9)": dict(fast=12, slow=26, signal=9,  rsi_period=14, rsi_upper=55, rsi_lower=45),
    "D C + RSI밴드 좁힘(52/48)":  dict(fast=12, slow=26, signal=9,  rsi_period=10, rsi_upper=52, rsi_lower=48),
    "E 공격적 단축(8/17/6, 53/47)": dict(fast=8,  slow=17, signal=6, rsi_period=9,  rsi_upper=53, rsi_lower=47),
}
COMMON = dict(bb_period=20, bb_std=2.0, bb_width_window_weeks=260, bb_width_warn_percentile=95,
              hist_upper=0.0, hist_lower=0.0, slope_window=3)


def load_all(cfg):
    out = {}
    for e in cfg["universe"]:
        if e["ticker"] in EXCLUDE:
            continue
        path = os.path.join("cache", f"{e['ticker'].replace('/', '_')}.parquet")
        if not os.path.exists(path):
            continue
        try:
            wk = D.load(e, cfg)
            out[e["ticker"]] = (wk, e)
        except Exception as ex:
            print(f"[skip] {e['ticker']}: {ex}")
    return out


def run_one_config(wk_cache, p_extra):
    p = {**COMMON, **p_extra}
    sigs = {}
    for tkr, (wk, e) in wk_cache.items():
        df = enrich(wk, p, e["kind"])
        sig = run_signals(df, p, {"kind": e["kind"], **e})
        if sig.empty:
            continue
        px = df["close"].reindex(sig.index)
        for h in HORIZONS:
            sig[f"fwd{h}"] = (px.shift(-h) / px - 1.0) * 100.0
        sigs[tkr] = (sig, df, e)
    return sigs


def turn_lag(sigs, direction, window_back=26, window_fwd=13):
    """freshly confirmed 매수/매도 시점이, 그 주변 실제 저점/고점보다 몇 주 늦었는가."""
    lags = []
    for tkr, (sig, df, e) in sigs.items():
        conf = sig["confirmed"]
        changed = (conf != conf.shift()) & (conf == direction)
        px = df["close"]
        for t in sig.index[changed]:
            lo = t - pd.Timedelta(weeks=window_back)
            hi = t + pd.Timedelta(weeks=window_fwd)
            window = px.loc[lo:hi]
            if len(window) < 5:
                continue
            ext_date = window.idxmin() if direction == "매수" else window.idxmax()
            lag_weeks = (t - ext_date).days / 7.0
            lags.append(lag_weeks)
    if not lags:
        return np.nan, np.nan, 0
    a = np.array(lags)
    return float(a.mean()), float(np.median(a)), len(a)


def pooled_excess_accuracy(sigs):
    out = {}
    for h in HORIZONS:
        up_all = n_all = buy_hits = buy_n = down_hits = sell_n = 0
        for tkr, (sig, df, e) in sigs.items():
            fwd = sig[f"fwd{h}"].dropna()
            up_all += int((fwd > 0).sum()); n_all += len(fwd)
            bm = sig.direction == "매수"; bv = sig.loc[bm, f"fwd{h}"].dropna()
            buy_hits += int((bv > 0).sum()); buy_n += len(bv)
            sm = sig.direction == "매도"; sv = sig.loc[sm, f"fwd{h}"].dropna()
            down_hits += int((sv < 0).sum()); sell_n += len(sv)
        base_up = up_all / n_all * 100 if n_all else np.nan
        buy_acc = buy_hits / buy_n * 100 if buy_n else np.nan
        sell_acc = down_hits / sell_n * 100 if sell_n else np.nan
        out[h] = {"buy_excess": buy_acc - base_up, "buy_n": buy_n,
                   "sell_excess": sell_acc - (100 - base_up), "sell_n": sell_n}
    return out


def annual_transitions(sigs):
    tot_chg, tot_years = 0, 0.0
    for tkr, (sig, df, e) in sigs.items():
        conf = sig["confirmed"]
        chg = int((conf != conf.shift()).sum()) - 1
        years = max((sig.index[-1] - sig.index[0]).days / 365.25, 1e-9)
        tot_chg += chg; tot_years += years
    return tot_chg / tot_years if tot_years else np.nan


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    wk_cache = load_all(cfg)
    print(f"{len(wk_cache)}개 종목 로드\n")

    print(f"{'구성':<32}{'매수지연(중앙값/평균,주)':<22}{'매도지연(중앙값/평균,주)':<22}"
          f"{'매수초과13주':<12}{'매도초과13주':<12}{'연간전환(종목평균)':<10}")
    print("-" * 130)
    results = {}
    for name, extra in CANDIDATES.items():
        sigs = run_one_config(wk_cache, extra)
        bmean, bmed, bn = turn_lag(sigs, "매수")
        smean, smed, sn = turn_lag(sigs, "매도")
        acc = pooled_excess_accuracy(sigs)
        freq = annual_transitions(sigs)
        print(f"{name:<32}{f'{bmed:.1f} / {bmean:.1f} (n={bn})':<22}"
              f"{f'{smed:.1f} / {smean:.1f} (n={sn})':<22}"
              f"{acc[13]['buy_excess']:+6.1f}%p n={acc[13]['buy_n']:<5}"
              f"{acc[13]['sell_excess']:+6.1f}%p n={acc[13]['sell_n']:<5}"
              f"{freq:5.2f}")
        results[name] = {"buy_lag_median": bmed, "buy_lag_mean": bmean, "buy_lag_n": bn,
                          "sell_lag_median": smed, "sell_lag_mean": smean, "sell_lag_n": sn,
                          "accuracy": acc, "annual_transitions": freq}

    with open("param_sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("\n결과 JSON 저장: param_sensitivity_results.json")


if __name__ == "__main__":
    main()
