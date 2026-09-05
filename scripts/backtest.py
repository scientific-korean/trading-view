"""
백테스트 = 신호 자체의 성질을 측정한다. 수익률 곡선(CAGR)이 아니다.
비중 결정이 재량이므로 수익률은 임의 가정에 좌우된다.

  1) 방향 정확도    매수 후 상승 / 매도 후 하락 비율 (기준 50%)
  2) 중립 구간      비중·지속·후행수익률
  3) 신호 지속/빈도  연간 확정전환 횟수
  4) BB width 분포  경고 임계치 결정용 백분위표
  5) 매도 지연      매도 확정 시점의 고점 대비 낙폭 (참고)
  6) 참고 이진곡선  strict(중립=0%) vs hold(중립=직전유지). 절대 성과 아님.

방향은 fast/slow/signal/RSI 임계값이라는 고정 파라미터로만 정해지는
순수 규칙 기반이라, 데이터로 학습하는 값이 없다.
그래서 학습/평가 구간을 나눌 필요가 없고 워크포워드도 필요 없다.
(세기 3단계를 폐기하면서 함께 사라진 복잡도)
"""
import sys, os, yaml, argparse
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import data as D
from core.indicators import enrich
from core.engine import run_signals

HORIZONS = [1, 4, 13]
MIN_N = 30       # 이보다 표본이 적으면 그 방향/기간 정확도는 참고만


def analyze(sig: pd.DataFrame, df: pd.DataFrame, name: str, kind: str):
    px = df["close"].reindex(sig.index)
    for h in HORIZONS:
        sig[f"fwd{h}"] = (px.shift(-h) / px - 1.0) * 100.0

    r = {"name": name, "kind": kind, "n": len(sig)}

    # 1) 방향 정확도 — 매수 후 상승 / 매도 후 하락 비율. 기준 50%.
    acc = {}
    for d, sign in (("매수", 1), ("매도", -1)):
        m = sig.direction == d
        for h in HORIZONS:
            v = sig.loc[m, f"fwd{h}"].dropna()
            acc[f"{d}{h}주"] = {"pct": float((v * sign > 0).mean() * 100) if len(v) else np.nan,
                               "n": int(len(v))}
    r["accuracy"] = acc

    # 2) 중립 구간
    nm = sig.direction == "중립"
    nr = {"비중%": float(nm.mean() * 100)}
    if nm.sum():
        g = sig.groupby((nm != nm.shift()).cumsum())
        rl = [len(x) for _, x in g if x["direction"].iloc[0] == "중립"]
        nr.update({"구간수": len(rl), "평균지속주": float(np.mean(rl)),
                   "최장주": int(max(rl))})
        for h in HORIZONS:
            nr[f"{h}주수익"] = float(sig.loc[nm, f"fwd{h}"].mean())
    r["neutral"] = nr

    # 3) 지속 / 빈도 (확정 방향 기준. 중립 진출입은 전환으로 세지 않는다)
    conf = sig["confirmed"]
    chg = conf != conf.shift()
    n_chg = int(chg.sum()) - 1
    years = max((sig.index[-1] - sig.index[0]).days / 365.25, 1e-9)
    runs = sig.groupby(chg.cumsum()).size()
    r["freq"] = {"전환수": n_chg, "연간확정전환": n_chg / years,
                 "평균지속주": float(runs.mean()), "최장지속주": int(runs.max())}

    # 4) BB width 분포
    w = sig["bb_width"].dropna()
    r["bbw"] = {f"p{q}": float(np.percentile(w, q)) for q in (50, 75, 90, 95, 99)} \
        if len(w) else {}
    r["bbw"]["max"] = float(w.max()) if len(w) else np.nan

    # 5) 매도 지연: 매도 확정 시점의 52주 고점 대비 낙폭
    dd = []
    roll_max = df["close"].rolling(52, min_periods=10).max().reindex(sig.index)
    for t in sig.index[(sig.direction == "매도") & chg]:
        if pd.notna(roll_max.get(t)):
            dd.append((px[t] / roll_max[t] - 1) * 100)
    r["sell_lag"] = {"n": len(dd), "평균낙폭": float(np.mean(dd)) if dd else np.nan,
                     "중앙값": float(np.median(dd)) if dd else np.nan}

    # 6) 참고 이진곡선. 절대 성과 아님, 중립 처리 감각용.
    ret = px.pct_change().fillna(0)
    bh = (1 + ret).cumprod()
    r["ref"] = {"BH배수": float(bh.iloc[-1]),
                "BH_MDD%": float(((bh / bh.cummax()) - 1).min() * 100)}
    for tag, base in (("strict", sig.direction), ("hold", sig["confirmed"])):
        pos = (base == "매수").astype(float).shift(1).fillna(0)
        cost = pos.diff().abs().fillna(0) * 0.001
        eq = (1 + pos * ret - cost).cumprod()
        r["ref"][f"{tag}배수"] = float(eq.iloc[-1])
        r["ref"][f"{tag}MDD%"] = float(((eq / eq.cummax()) - 1).min() * 100)
    return r


def show(r):
    print(f"\n{'='*70}\n■ {r['name']}  ({r['kind']}, 주봉 {r['n']}개)\n{'='*70}")
    if r["n"] < 100:
        print("  ⚠ 표본 부족. 통계적 판단 불가. 참고만.")

    print("\n[방향 정확도 %  기준 50%]")
    for k, v in r["accuracy"].items():
        flag = "" if v["n"] >= MIN_N else "  (n부족)"
        pct = f"{v['pct']:.1f}" if v["pct"] == v["pct"] else "—"
        print(f"  {k:<8} {pct:>6}%   n={v['n']}{flag}")

    print("[중립 구간]", {k: (f"{v:.2f}" if isinstance(v, float) else v)
                        for k, v in r["neutral"].items()})
    print("[신호 빈도]", {k: (f"{v:.2f}" if isinstance(v, float) else v)
                        for k, v in r["freq"].items()})
    print("[BB width 분포]", {k: f"{v:.3f}" for k, v in r["bbw"].items() if v == v})
    print("[매도 지연]", {k: (f"{v:.1f}" if isinstance(v, float) and v == v else v)
                        for k, v in r["sell_lag"].items()})
    print("[참고 이진곡선]", {k: f"{v:.2f}" for k, v in r["ref"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", nargs="*", help="티커 필터")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    p = dict(cfg["params"])
    results = []

    for e in cfg["universe"]:
        if a.only and e["ticker"] not in a.only:
            continue
        try:
            wk = D.load(e, cfg)
            df = enrich(wk, p, e["kind"])
            sig = run_signals(df, p, {"kind": e["kind"], **e})
            if sig.empty:
                print(f"[skip] {e['name']}: 신호 없음"); continue
            r = analyze(sig, df, e["name"], e["kind"])
            show(r); results.append(r)
        except Exception as ex:
            print(f"[error] {e['name']}: {ex}")

    print(f"\n{'='*70}\n■ 종합 - 방향 정확도 (풀링)\n{'='*70}")
    if not results:
        print("평가 가능한 종목 없음."); return

    # 종목을 가로질러 관측치를 풀링. 종목 수가 아니라 표본 수 기준 가중.
    for h in HORIZONS:
        for d, sign in (("매수", 1), ("매도", -1)):
            key = f"{d}{h}주"
            n_tot = sum(r["accuracy"][key]["n"] for r in results)
            if n_tot == 0:
                continue
            hit = sum(r["accuracy"][key]["pct"] / 100 * r["accuracy"][key]["n"]
                      for r in results if r["accuracy"][key]["n"])
            pct = hit / n_tot * 100
            # 이항분포 근사 표준오차. 50% 기준 z 값.
            se = 100 * (0.25 / n_tot) ** 0.5
            z = (pct - 50) / se if se else 0
            sig_mark = "**" if abs(z) >= 2 else ("*" if abs(z) >= 1 else "")
            print(f"  {key:<8} {pct:5.1f}%   n={n_tot:<5}  "
                  f"z={z:+.1f}{sig_mark}")

    print("\n→ 기준 50%. z ≥ 2(또는 ≤ -2) 면 대략 95% 수준에서 무작위와 구분됨(*)."
          "\n  단, 후행수익률은 주간 관측치가 겹쳐 자기상관이 있고 종목 간 상관도"
          "\n  있어 이 z 값은 엄밀한 유의성 검정이 아니라 방향을 보는 참고치다.")


if __name__ == "__main__":
    main()
