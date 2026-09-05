"""
매주 금요일 종가 후 실행.
  1) 데이터 수집 → 주봉 (미완성 주 제외)
  2) 지표 계산 → generate_signal 호출
  3) state.json 이월
  4) 텔레그램 발송 + docs/index.html 생성
"""
import os, sys, json, unicodedata, datetime as dt
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

# 텔레그램 <pre>(고정폭 글꼴) 메시지의 컬럼이 안 맞는 문제 수정용(2026-09).
# 한글(및 기타 동아시아 문자)은 고정폭 글꼴에서 폭 2로 그려지는데, 파이썬의
# f"{s:<12}" 같은 정렬은 "문자 개수" 기준이라 한글 비중이 다른 두 문자열을
# 나란히 두면 화면상 폭이 서로 달라져 줄이 어긋난다. unicodedata의
# East Asian Width 속성(W/F = 폭 2, 그 외 = 폭 1)으로 "화면상 폭"을 직접
# 계산해서 자르고/채운다(_vpad/_vtrunc). NAME_W=14, LABEL_W=15는 현재
# config.yaml의 종목명·상태라벨 중 가장 넓은 것("마이크로소프트"/
# "버크셔해서웨이"=14, "중립(직전 매수)" 류=15) 기준이며, 나중에 이보다 긴
# 이름이 추가되면 잘려서 표시된다(줄 자체는 안 깨짐).
#
# 화면폭 기준으로 맞춰도 종목명+상태+RSI+OSC를 한 줄에 다 넣으면 모바일
# 화면 폭 자체를 넘어서서 중간에 줄바꿈이 일어나 오히려 더 지저분해진다
# (2026-09 제보) — 그래서 아래 main()에서 종목당 2줄(이름+상태 / RSI+OSC)로
# 나눠 각 줄 길이를 줄인다.
NAME_W, LABEL_W = 14, 15


def _vwidth(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _vtrunc(s: str, width: int) -> str:
    out, w = "", 0
    for ch in s:
        cw = _vwidth(ch)
        if w + cw > width:
            break
        out += ch
        w += cw
    return out


def _vpad(s: str, width: int) -> str:
    s = _vtrunc(s, width)
    return s + " " * max(0, width - sum(_vwidth(c) for c in s))


def main(cfg_path="config.yaml", no_cache=True):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    p = dict(cfg["params"])
    states = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}

    payload, lines, errs = [], [], []
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
            warn = " ⚠" if dec["flags"]["vol_warning"] else ""
            fmt = lambda v, d=2: f"{v:.{d}f}" if v is not None else "—"
            # OSC 컬럼: 예전엔 osc_line(PPO/MACD 원값)을 보여줬는데, 실제 매수/매도
            # 판정에 쓰이는 값은 히스토그램(osc_hist, hist_upper/lower와 비교되는 값)이라
            # 서로 다른 숫자였다(예: osc_line 3.658 vs 판정에 쓰인 osc_hist 0.190).
            # RSI처럼 "판정에 실제로 쓰인 값"을 그대로 보여주도록 osc_hist로 교체.
            #
            # 1줄에 종목명+상태+RSI+OSC를 다 넣으면(문자폭 기준으로 정렬해도) 모바일
            # 화면 폭을 넘어서서 중간에 줄바꿈이 일어나 정렬이 오히려 더 깨졌다(2026-09
            # 제보). 종목명+상태를 1번째 줄, RSI/OSC를 들여쓴 2번째 줄로 나눠 각 줄
            # 길이를 줄인다 — 종목명/상태는 한글 위주라 화면폭 기준 _vpad, RSI/OSC는
            # 숫자·기호뿐이라(ASCII는 폭이 항상 1이라 문자 개수 정렬로 충분) 그냥
            # :>width로 정렬한다.
            name_line = f"{mark}{_vpad(e['name'], NAME_W)} {_vtrunc(lbl, LABEL_W)}"
            detail_line = f"   RSI {fmt(dec['rsi'],1):>5} OSC {fmt(dec['osc_hist'],3):>7}{warn}"
            lines.append(name_line + "\n" + detail_line)
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

    changed = [l for l in lines if l.startswith("◆")]
    msg = f"<b>주봉 신호 {asof}</b>\n"
    msg += (f"\n<b>전환 {len(changed)}건</b>\n<pre>" + "\n".join(changed) + "</pre>\n"
            if changed else "\n전환 없음\n")
    msg += "\n<pre>" + "\n".join(lines) + "</pre>"
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
