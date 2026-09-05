"""
세 가지 후속 검증 (이전 세션에서 도출된 to-do):
  1) Buy&Hold 기준선(무조건부 상승/하락 확률) 대비 매수/매도 정확도
  2) 종목군(지수/환율/원자재/금리/개별)별 분해
  3) 2008·2020·2022 하락장 구간만 별도로 매도 정확도 재검증
  + 부록: "매도 신호가 하락이 아니라 반등을 예고" 가설 정량 검증

네트워크 불필요 — cache/ 의 parquet만 사용.
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

BEAR_WINDOWS = {
    "2008 금융위기": ("2007-10-01", "2009-03-31"),
    "2020 코로나 폭락": ("2020-02-01", "2020-04-30"),
    "2022 하락장": ("2022-01-01", "2022-10-31"),
}


def load_universe(cfg):
    out = []
    for e in cfg["universe"]:
        path = os.path.join("cache", f"{e['ticker'].replace('/', '_')}.parquet")
        if not os.path.exists(path):
            print(f"[skip] {e['name']}: 캐시 없음 ({e['ticker']})")
            continue
        out.append(e)
    return out


def build_signals(cfg):
    """전 종목 신호 생성. {ticker: (sig_df, price_df, entry)} 반환."""
    p = dict(cfg["params"])
    result = {}
    for e in load_universe(cfg):
        try:
            wk = D.load(e, cfg)
            df = enrich(wk, p, e["kind"])
            sig = run_signals(df, p, {"kind": e["kind"], **e})
            if sig.empty:
                continue
            px = df["close"].reindex(sig.index)
            for h in HORIZONS:
                sig[f"fwd{h}"] = (px.shift(-h) / px - 1.0) * 100.0
            result[e["ticker"]] = (sig, df, e)
        except Exception as ex:
            print(f"[error] {e['name']}: {ex}")
    return result


def pooled_rate(rows, col_mask_fn, ret_col, sign):
    """rows: [(sig, n_weight_ignored)] 리스트. 부호조건 만족 비율(%) 과 n 반환."""
    hits, n = 0, 0
    for sig in rows:
        m = col_mask_fn(sig)
        v = sig.loc[m, ret_col].dropna()
        if len(v) == 0:
            continue
        hits += int((v * sign > 0).sum())
        n += len(v)
    return (hits / n * 100 if n else np.nan), n


def zscore(pct, n, base=50.0):
    if not n:
        return np.nan
    se = 100 * (0.25 / n) ** 0.5
    return (pct - base) / se if se else np.nan


# ------------------------------------------------------------------
# 1) Buy&Hold 기준선 대비 정확도
# ------------------------------------------------------------------
def task1_baseline_comparison(sigs):
    print("\n" + "=" * 78)
    print("■ 1) Buy&Hold 기준선(무조건부 상승/하락 확률) 대비 매수/매도 정확도")
    print("=" * 78)
    print("  기준선 = 신호 상태와 무관하게 '아무 주'에나 있었을 때 h주 후 상승/하락 확률.")
    print("  자산 자체의 장기 드리프트(우상향 추세)를 걷어낸 비교치.\n")

    rows = []
    for h in HORIZONS:
        all_up_hits, all_up_n = 0, 0
        buy_hits, buy_n = 0, 0
        sell_down_hits, sell_n = 0, 0
        for tkr, (sig, df, e) in sigs.items():
            fwd = sig[f"fwd{h}"].dropna()
            all_up_hits += int((fwd > 0).sum())
            all_up_n += len(fwd)

            bm = sig.direction == "매수"
            bv = sig.loc[bm, f"fwd{h}"].dropna()
            buy_hits += int((bv > 0).sum()); buy_n += len(bv)

            sm = sig.direction == "매도"
            sv = sig.loc[sm, f"fwd{h}"].dropna()
            sell_down_hits += int((sv < 0).sum()); sell_n += len(sv)

        base_up = all_up_hits / all_up_n * 100 if all_up_n else np.nan
        base_down = 100 - base_up
        buy_acc = buy_hits / buy_n * 100 if buy_n else np.nan
        sell_acc = sell_down_hits / sell_n * 100 if sell_n else np.nan

        rows.append({
            "h": h,
            "BH_상승기준선%": base_up, "매수정확도%": buy_acc, "매수_초과%p": buy_acc - base_up,
            "매수n": buy_n, "매수_z(vs50)": zscore(buy_acc, buy_n, 50),
            "매수_z(vsBH)": zscore(buy_acc, buy_n, base_up),
            "BH_하락기준선%": base_down, "매도정확도%": sell_acc, "매도_초과%p": sell_acc - base_down,
            "매도n": sell_n, "매도_z(vs50)": zscore(sell_acc, sell_n, 50),
            "매도_z(vsBH)": zscore(sell_acc, sell_n, base_down),
        })

        print(f"[{h}주 후]")
        print(f"  BH 상승기준선 {base_up:5.1f}%  |  매수신호 정확도 {buy_acc:5.1f}%  "
              f"(초과 {buy_acc-base_up:+5.1f}%p, n={buy_n}, z_vs50={zscore(buy_acc,buy_n,50):+.1f}, "
              f"z_vsBH={zscore(buy_acc,buy_n,base_up):+.1f})")
        print(f"  BH 하락기준선 {base_down:5.1f}%  |  매도신호 정확도 {sell_acc:5.1f}%  "
              f"(초과 {sell_acc-base_down:+5.1f}%p, n={sell_n}, z_vs50={zscore(sell_acc,sell_n,50):+.1f}, "
              f"z_vsBH={zscore(sell_acc,sell_n,base_down):+.1f})\n")
    return rows


# ------------------------------------------------------------------
# 2) 종목군별 분해
# ------------------------------------------------------------------
def task2_group_breakdown(sigs):
    print("\n" + "=" * 78)
    print("■ 2) 종목군별 분해 (지수 / 환율 / 원자재 / 금리 / 개별)")
    print("=" * 78)

    groups = {}
    for tkr, (sig, df, e) in sigs.items():
        groups.setdefault(e["group"], []).append((tkr, sig, df, e))

    out = {}
    for g, items in groups.items():
        tickers = [i[0] for i in items]
        print(f"\n--- {g}  (종목: {', '.join(tickers)}) ---")
        gr = {"tickers": tickers, "by_h": {}}
        for h in HORIZONS:
            buy_hits = buy_n = down_hits = sell_n = up_all = n_all = 0
            for tkr, sig, df, e in items:
                fwd = sig[f"fwd{h}"].dropna()
                up_all += int((fwd > 0).sum()); n_all += len(fwd)
                bm = sig.direction == "매수"
                bv = sig.loc[bm, f"fwd{h}"].dropna()
                buy_hits += int((bv > 0).sum()); buy_n += len(bv)
                sm = sig.direction == "매도"
                sv = sig.loc[sm, f"fwd{h}"].dropna()
                down_hits += int((sv < 0).sum()); sell_n += len(sv)
            base_up = up_all / n_all * 100 if n_all else np.nan
            buy_acc = buy_hits / buy_n * 100 if buy_n else np.nan
            sell_acc = down_hits / sell_n * 100 if sell_n else np.nan
            gr["by_h"][h] = {"base_up": base_up, "buy_acc": buy_acc, "buy_n": buy_n,
                              "sell_acc": sell_acc, "sell_n": sell_n,
                              "base_down": 100 - base_up}
            print(f"  {h:>2}주  BH상승기준 {base_up:5.1f}%  "
                  f"매수 {buy_acc:5.1f}%(초과{buy_acc-base_up:+5.1f}%p,n={buy_n:<4}) "
                  f"매도 {sell_acc:5.1f}%(초과{sell_acc-(100-base_up):+5.1f}%p,n={sell_n})")
        out[g] = gr

    print("\n※ 유의: 개별주 중 NVDA·MSFT·GOOGL·AMZN·META·AVGO·TSM 7종은 상관이 매우 높아"
          "\n  '개별' 그룹의 표본 수가 독립 관측치 수를 과장한다 (사실상 소수 매크로 베팅).")
    return out


# ------------------------------------------------------------------
# 3) 2008·2020·2022 하락장 구간 매도 정확도 재검증
# ------------------------------------------------------------------
def task3_bear_market_recheck(sigs):
    print("\n" + "=" * 78)
    print("■ 3) 하락장 구간(2008·2020·2022)만 별도 매도 정확도 재검증")
    print("=" * 78)

    out = {}
    for label, (start, end) in BEAR_WINDOWS.items():
        print(f"\n--- {label} ({start} ~ {end}) ---")
        covering = [tkr for tkr, (sig, df, e) in sigs.items()
                    if sig.index.min() <= pd.Timestamp(start)]
        print(f"  해당 구간 데이터 보유 종목 {len(covering)}/{len(sigs)}개: "
              f"{', '.join(covering) if len(covering) <= 12 else ', '.join(covering[:12])+' 외'}")

        row = {"window": [start, end], "n_assets_covering": len(covering), "by_h": {}}
        for h in HORIZONS:
            down_hits = sell_n = 0
            base_down_hits = base_n = 0
            per_asset = []
            for tkr, (sig, df, e) in sigs.items():
                win = sig.loc[start:end]
                if win.empty:
                    continue
                fwd_all = win[f"fwd{h}"].dropna()
                base_down_hits += int((fwd_all < 0).sum()); base_n += len(fwd_all)

                sm = win.direction == "매도"
                sv = win.loc[sm, f"fwd{h}"].dropna()
                if len(sv):
                    down_hits += int((sv < 0).sum()); sell_n += len(sv)
                    per_asset.append((e["name"], len(sv), float((sv < 0).mean() * 100)))

            sell_acc = down_hits / sell_n * 100 if sell_n else np.nan
            base_down = base_down_hits / base_n * 100 if base_n else np.nan
            row["by_h"][h] = {"sell_acc": sell_acc, "sell_n": sell_n,
                               "base_down": base_down, "base_n": base_n}
            print(f"  {h:>2}주  매도→하락 정확도 {sell_acc:5.1f}%  (n={sell_n})   "
                  f"vs 구간내 무조건부 하락기준선 {base_down:5.1f}%  (n={base_n})   "
                  f"초과 {sell_acc-base_down:+5.1f}%p" if sell_n else
                  f"  {h:>2}주  매도 신호 표본 없음")
        out[label] = row
    return out


# ------------------------------------------------------------------
# 부록) "매도 신호가 반등을 예고" 가설 정량 검증
# ------------------------------------------------------------------
def appendix_rebound_hypothesis(sigs):
    print("\n" + "=" * 78)
    print("■ 부록) '매도 신호 = 하락이 아니라 반등 예고' 가설 검증")
    print("=" * 78)

    # (a) 매도 구간 전체 vs 매도 확정 직후(changed) vs 매도 지속 중, 평균 후행수익률
    for h in HORIZONS:
        all_ret, fresh_ret, hold_ret = [], [], []
        for tkr, (sig, df, e) in sigs.items():
            m = sig.direction == "매도"
            all_ret += sig.loc[m, f"fwd{h}"].dropna().tolist()
            conf = sig["confirmed"]
            changed = (conf != conf.shift()) & (conf == "매도")
            fresh = m & changed
            fresh_ret += sig.loc[fresh, f"fwd{h}"].dropna().tolist()
            hold_ret += sig.loc[m & ~changed, f"fwd{h}"].dropna().tolist()

        def stat(a):
            a = np.array(a)
            if len(a) == 0:
                return (np.nan, np.nan, 0, np.nan)
            return (a.mean(), np.median(a), len(a), float((a > 0).mean() * 100))

        am, amed, an, aup = stat(all_ret)
        fm, fmed, fn, fup = stat(fresh_ret)
        hm, hmed, hn, hup = stat(hold_ret)
        print(f"\n[{h}주 후 수익률 | 매도 상태 기준]")
        print(f"  전체 매도구간   평균 {am:+5.2f}%  중앙값 {amed:+5.2f}%  n={an:<5} 상승비율 {aup:5.1f}%")
        print(f"  ├ 매도 '확정 직후'(hysteresis 확인 시점)  평균 {fm:+5.2f}%  중앙값 {fmed:+5.2f}%  "
              f"n={fn:<5} 상승비율 {fup:5.1f}%")
        print(f"  └ 매도 '지속 중'                          평균 {hm:+5.2f}%  중앙값 {hmed:+5.2f}%  "
              f"n={hn:<5} 상승비율 {hup:5.1f}%")

    # (b) 매도 확정 시점의 52주 고점 대비 낙폭(=이미 진행된 하락폭) 분위수별,
    #     그 시점 이후 forward return 이 어떻게 달라지는지 — 이미 많이 빠졌을수록 반등하는가?
    print("\n[매도 '확정' 시점의 기누적낙폭(52주 고점 대비) 분위수별 향후 13주 수익률]")
    recs = []
    for tkr, (sig, df, e) in sigs.items():
        px = df["close"].reindex(sig.index)
        roll_max = df["close"].rolling(52, min_periods=10).max().reindex(sig.index)
        conf = sig["confirmed"]
        changed = (conf != conf.shift()) & (conf == "매도")
        for t in sig.index[changed]:
            if pd.notna(roll_max.get(t)) and pd.notna(sig.loc[t, "fwd13"]):
                dd = (px[t] / roll_max[t] - 1) * 100
                recs.append((dd, sig.loc[t, "fwd13"]))
    if recs:
        ddf = pd.DataFrame(recs, columns=["dd_at_sell", "fwd13"])
        ddf["bucket"] = pd.qcut(ddf["dd_at_sell"], 3, labels=["낙폭 큼(더 빠짐)", "중간", "낙폭 작음(초입)"])
        for b, g in ddf.groupby("bucket", observed=True):
            print(f"  {b:<16} 확정시점 낙폭평균 {g.dd_at_sell.mean():+6.1f}%   "
                  f"→ 이후13주 평균수익 {g.fwd13.mean():+5.2f}%  중앙값 {g.fwd13.median():+5.2f}%  "
                  f"상승비율 {(g.fwd13>0).mean()*100:5.1f}%  n={len(g)}")
        corr = ddf["dd_at_sell"].corr(ddf["fwd13"])
        print(f"  → 확정시점 낙폭과 향후13주수익의 상관계수: {corr:+.3f} "
              f"({'낙폭이 클수록(더 많이 빠졌을수록) 반등폭도 크다' if corr < -0.05 else '뚜렷한 관계 없음' if abs(corr)<=0.05 else '낙폭 클수록 반등도 작다(직관과 반대)'})")
    return recs


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    print("신호 생성 중 (캐시만 사용, 네트워크 없음)...")
    sigs = build_signals(cfg)
    print(f"완료: {len(sigs)}개 종목 신호 생성됨\n")

    r1 = task1_baseline_comparison(sigs)
    r2 = task2_group_breakdown(sigs)
    r3 = task3_bear_market_recheck(sigs)
    r4 = appendix_rebound_hypothesis(sigs)

    with open("decompose_results.json", "w", encoding="utf-8") as f:
        json.dump({"task1": r1, "task2": r2, "task3": r3}, f, ensure_ascii=False, indent=2, default=str)
    print("\n\n결과 JSON 저장: decompose_results.json")


if __name__ == "__main__":
    main()
