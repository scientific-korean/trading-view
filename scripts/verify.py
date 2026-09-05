"""합성 데이터로 로직 검증. 네트워크 불필요."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.indicators import enrich, rsi_wilder, macd_ppo, bb_width_percentile
from core.engine import run_signals
from core.signals import generate_signal

P = dict(fast=19, slow=39, signal=14, rsi_period=14, rsi_upper=55, rsi_lower=45,
         bb_period=20, bb_std=2.0, bb_width_window_weeks=260, bb_width_warn_percentile=95,
         hist_upper=0.0, hist_lower=0.0, slope_window=3)

rng = np.random.default_rng(42)
n = 600
idx = pd.date_range("2013-01-04", periods=n, freq="W-FRI")
px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0015, 0.025, n))), index=idx)
df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99, "close": px})

print("=" * 62)
# 1. RSI Wilder — 교과서 예제로 검증
ref = pd.Series([44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,45.84,46.08,
                 45.89,46.03,45.61,46.28,46.28,46.00,46.03,46.41,46.22,45.64,
                 46.21,46.25,45.71,46.45,45.78,45.35,44.03,44.18,44.22,44.57])
r = rsi_wilder(ref, 14)
print(f"1. RSI(Wilder) 검증  계산={r.iloc[14]:.2f}  기대 70.46  "
      f"{'OK' if abs(r.iloc[14]-70.46) < 0.1 else 'FAIL'}")
print(f"   범위 0~100 유지: {'OK' if (r.dropna().between(0,100)).all() else 'FAIL'}")

# 2. PPO와 MACD의 크로스 시점이 동일한가
ppo = macd_ppo(px, 19, 39, 14, "price")
mac = macd_ppo(px, 19, 39, 14, "rate")
sp, sm = np.sign(ppo.osc_hist), np.sign(mac.osc_hist)
agree = (sp == sm).mean() * 100
zline = (np.sign(ppo.osc_line) == np.sign(mac.osc_line)).mean() * 100
print(f"2. PPO vs MACD")
print(f"   라인 0선 부호 일치 {zline:.1f}%  ← 항상 100%여야 함 "
      f"{'OK' if zline == 100 else 'FAIL'}")
print(f"   히스토그램 부호 일치 {agree:.1f}%  ← 100%가 아닌 것이 정상")
print(f"   (시그널선을 각자 EMA로 만들므로 크로스가 1봉 어긋날 수 있음)")
print(f"   스케일  MACD={mac.osc_line.iloc[-1]:.2f}  PPO={ppo.osc_line.iloc[-1]:.3f}%")

# 3. 룩어헤드 없음 — 뒤쪽 데이터를 바꿔도 앞쪽 신호가 안 변해야 함
d1 = enrich(df, P, "price")
s1 = run_signals(d1, P, {"kind": "price"})
df2 = df.copy(); df2.iloc[-50:] *= 3.0
s2 = run_signals(enrich(df2, P, "price"), P, {"kind": "price"})
k = min(len(s1), len(s2)) - 50
same = s1.direction.iloc[:k].equals(s2.direction.iloc[:k])
print(f"3. 룩어헤드 차단: {'OK' if same else 'FAIL'}")

# 4. state 순수성 — 원본 dict 훼손 없음
st = {"direction": "매수", "rsi_zone": "bull", "osc_zone": "bull"}
snap = dict(st)
generate_signal(d1, st, P, {"kind": "price"})
print(f"4. state 불변성: {'OK' if st == snap else 'FAIL'}")

# 5. JSON 직렬화
import json
_, ns = generate_signal(d1, {}, P, {"kind": "price"})
try:
    json.dumps(ns); print("5. state JSON 직렬화: OK")
except TypeError as ex:
    print(f"5. state JSON 직렬화: FAIL {ex}")

# 6. 히스테리시스 동작 — RSI 45~55 구간에서 상태 유지
z = []
for v in [60, 52, 48, 46, 44, 50, 56]:
    prev = z[-1] if z else "neutral"
    from core.signals import _zone
    z.append(_zone(v, prev, 55, 45))
print(f"6. 히스테리시스: {z}")
print(f"   중간구간 유지: {'OK' if z[1]==z[2]==z[3]=='bull' and z[4]=='bear' else 'FAIL'}")

# 7. 금리 kind — 마이너스 금리에서 PPO 폭발 방지
neg = pd.Series(np.linspace(-0.5, 1.5, 200), index=pd.date_range("2020-01-03", periods=200, freq="W-FRI"))
ndf = pd.DataFrame({"open": neg, "high": neg, "low": neg, "close": neg})
rate = enrich(ndf, P, "rate")
fin = np.isfinite(rate.osc_line.dropna()).all()
print(f"7. 마이너스 금리 안정성(rate): {'OK' if fin else 'FAIL'}")
bad = enrich(ndf, P, "price").osc_line.dropna()
print(f"   같은 데이터에 price 적용 시 최대 |PPO| = {bad.abs().max():.0f}  ← rate 필수 이유")

# 8. 신호 요약
print(f"\n8. 합성 600주 신호 분포:\n{s1.direction.value_counts().to_string()}")
chg = (s1.direction != s1.direction.shift()).sum() - 1
print(f"   전환 {chg}회 / {len(s1)/52:.1f}년 = 연 {chg/(len(s1)/52):.1f}회")
print(f"   BB width 분포 p50={s1.bb_width.quantile(.5):.3f} "
      f"p95={s1.bb_width.quantile(.95):.3f} max={s1.bb_width.max():.3f}")

# 9. BB width 백분위 경고 (2026-09, 절대 임계치 → 자산별 상대 백분위로 변경)
wp = bb_width_percentile(d1["bb_width"], P["bb_width_window_weeks"])
in_range = wp.dropna().between(0, 100).all()
warm = wp.iloc[:P["bb_width_window_weeks"] - 1].isna().all()  # 윈도우 차기 전엔 전부 NaN
# 정의상 상위 (100-p)%가 경고 대상이어야 하므로, 유효 구간에서 경고 비율이
# 대략 그 근처(합성 데이터라 정확히 5%는 아니고 근사치)인지만 느슨하게 확인.
warn_rate = (wp.dropna() >= P["bb_width_warn_percentile"]).mean() * 100
ok9 = in_range and warm and 0 < warn_rate < 15
print(f"\n9. BB width 백분위 경고: 범위0~100 {'OK' if in_range else 'FAIL'}  "
      f"윈도우 전 NaN {'OK' if warm else 'FAIL'}  "
      f"경고비율 {warn_rate:.1f}% {'OK' if 0 < warn_rate < 15 else 'FAIL'}  "
      f"→ 종합 {'OK' if ok9 else 'FAIL'}")
print("=" * 62)
