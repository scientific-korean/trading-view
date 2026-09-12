"""
매주 금요일 종가 후 실행.
  1) 데이터 수집 → 주봉 (미완성 주 제외)
  2) 지표 계산 → generate_signal 호출
  3) state.json 이월
  4) 텔레그램 발송 + docs/index.html 생성
"""
import os, sys, json, datetime as dt
import yaml
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# 로컬 실행용 .env 로드 (TG_TOKEN/TG_CHAT_ID/ECOS_API_KEY). 파일이 없으면 조용히
# 넘어가고 기존 OS 환경변수를 그대로 쓴다 — GitHub Actions에서는 .env가 없고
# secrets가 이미 os.environ에 들어있으므로 이 줄은 아무 영향 없음.
# 이미 설정된 OS 환경변수를 .env 값이 덮어쓰지 않도록 override=False(기본값) 유지.
load_dotenv(os.path.join(_ROOT, ".env"))

from core import data as D
from core.indicators import enrich
from core.signals import generate_signal
from core.chart import build_html

STATE = "state.json"
OUT = "docs/index.html"

# 텔레그램 메시지 포맷(2026-09~): 종목마다 구분선+종목명+상세, 3줄짜리 "박스"로 표시한다.
#   ======================
#   메타
#   : 매수 / OSC -1.294 / RSI 23.2
# 이전엔 종목명·상태·RSI·OSC를 한 줄(또는 들여쓴 2줄)에 몰아넣고 동아시아 문자폭
# (한글=2, 영문/숫자=1)까지 보정해가며 여러 종목의 컬럼을 맞추려 했는데, 그래도
# 모바일 폭을 넘기면 중간에 강제 줄바꿈이 나 정렬이 깨졌다(2026-09 제보 다수).
# 종목마다 줄 자체를 통째로 분리하는 이 포맷에서는 애초에 "여러 종목이 세로로
# 정렬돼야 한다"는 제약이 없어져 그런 폭 보정이 더 이상 필요 없다.
SEP = "=" * 22


def main(cfg_path="config.yaml", no_cache=True):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    p = dict(cfg["params"])
    states = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}

    payload, lines, changed_lines, errs = [], [], [], []
    for e in cfg["universe"]:
        try:
            wk = D.load(e, cfg, use_cache=not no_cache)
            df = enrich(wk, p, e["kind"])
            prev = states.get(e["ticker"])

            if not prev:
                # 콜드스타트: 직전 봉까지 히스토리를 순차 재생해 state 복원.
                # 없으면 최초 실행 시 전 종목이 '관망'으로 뜬다.
                from core.engine import run_signals
                _, prev = run_signals(df.iloc[:-1], p, {"kind": e["kind"], **e},
                                      return_state=True)

            dec, st = generate_signal(df, prev, p, {"kind": e["kind"], **e})
            states[e["ticker"]] = st
            payload.append((e, df, dec))

            lbl = dec["direction"]
            if e["kind"] == "rate":
                lbl = {"매수": "상승", "매도": "하락"}.get(lbl, lbl)
            if dec["direction"] == "중립" and dec["confirmed"]:
                c = dec["confirmed"]
                if e["kind"] == "rate":
                    c = {"매수": "상승", "매도": "하락"}[c]
                lbl = f"중립(직전 {c})"
            mark = "◆" if dec["changed"] else ("·" if dec["neutral_edge"] else " ")
            fmt = lambda v, d=2: f"{v:.{d}f}" if v is not None else "—"
            # OSC: 예전엔 osc_line(PPO/MACD 원값)을 보여줬는데, 실제 매수/매도 판정에
            # 쓰이는 값은 히스토그램(osc_hist, hist_upper/lower와 비교되는 값)이라 서로
            # 다른 숫자였다(예: osc_line 3.658 vs 판정에 쓰인 osc_hist 0.190). RSI처럼
            # "판정에 실제로 쓰인 값"을 그대로 보여주도록 osc_hist를 쓴다.
            #
            # BB width 변동성 경고(⚠)는 텔레그램 메시지에서는 뺐다(2026-09, 사용자
            # 요청 — 정보가 늘어나 복잡해짐). 대시보드 차트의 BB width 패널에는
            # 계속 표시되므로 정보 자체가 없어지는 건 아니다.
            block = (f"{SEP}\n{mark}{e['name']}\n"
                     f": {lbl} / OSC {fmt(dec['osc_hist'],3)} / RSI {fmt(dec['rsi'],1)}")
            lines.append(block)
            if dec["changed"]:
                changed_lines.append(block)
        except Exception as ex:
            errs.append(f"{e['name']}: {ex}")

    # payload[0](유니버스 첫 종목)의 날짜를 그대로 썼더니, 그 종목 하나만
    # 데이터가 막혀도(2026-09 ^KS200 사례) 전체 리포트 날짜가 틀리게 찍혔다.
    # 다수결(최빈값)로 바꿔서 소수 종목의 결측에 안 흔들리게 한다.
    if payload:
        dates = [df.index[-1].date() for _, df, _ in payload]
        asof = max(set(dates), key=dates.count).isoformat()
    else:
        asof = str(dt.date.today())

    # config.yaml에서 완전히 빠진 티커의 옛 state는 정리한다(2026-09: ^KS200→^KS11
    # 교체 후에도 "^KS200" 항목이 state.json에 계속 남아있던 문제). 이번 실행에서
    # 일시적으로 fetch가 실패한 티커(여전히 유니버스엔 있음)는 여기 안 걸리므로
    # 그 상태는 그대로 보존된다 — 유니버스에서 아예 제거된 티커만 지워진다.
    current_tickers = {e["ticker"] for e in cfg["universe"]}
    stale = set(states) - current_tickers
    if stale:
        # 오류가 아니라 정리 로그라 텔레그램 메시지(errs)엔 안 넣고 콘솔/CI 로그에만 남긴다.
        print(f"[정리] state.json: 유니버스에 없는 옛 항목 제거 → {sorted(stale)}")
    states = {k: v for k, v in states.items() if k in current_tickers}

    json.dump(states, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    os.makedirs("docs", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(build_html(payload, p, asof))

    msg = f"<b>주봉 신호 {asof}</b>\n"
    if changed_lines:
        msg += (f"\n<b>전환 {len(changed_lines)}건</b>\n<pre>"
                + "\n".join(changed_lines) + "\n" + SEP + "</pre>\n")
    else:
        msg += "\n전환 없음\n"
    msg += "\n<pre>" + "\n".join(lines) + "\n" + SEP + "</pre>"
    if errs:
        msg += "\n<b>오류</b>\n<pre>" + "\n".join(errs) + "</pre>"

    print(msg)
    send(msg)
    return 0


def send(text: str):
    tok, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT_ID")
    if not (tok and chat):
        print("[TG 미설정 - 발송 생략]"); return
    import requests
    r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": text[:4000],
                            "parse_mode": "HTML"}, timeout=20)
    print("[TG]", r.status_code)


if __name__ == "__main__":
    sys.exit(main())
