"""
=============================================================
  신호 설계 파일  ―  여기만 고치면 된다
=============================================================
generate_signal 은 순수 함수여야 한다.
  - 파일 입출력 / 네트워크 / 전역변수 / 현재시각 조회 금지
  - 같은 입력에는 항상 같은 출력
백테스트와 라이브가 이 함수를 똑같이 호출하므로,
이 규약이 깨지면 백테스트 결과가 실제와 달라진다.

state 는 JSON 직렬화 가능해야 한다 (float/str/int/bool/None 만).
numpy 타입은 float() 로 변환해서 담을 것.
"""


def _zone(value, prev_zone, upper, lower):
    """히스테리시스 존 판정. 중간 구간에서는 직전 존을 유지."""
    if value is None:
        return prev_zone
    if value > upper:
        return "bull"
    if value < lower:
        return "bear"
    return prev_zone


def generate_signal(df, state, params, meta=None):
    """
    Parameters
    ----------
    df : DataFrame  지표까지 계산된 주봉. 마지막 행이 평가 대상.
                    미래 데이터는 애초에 포함되지 않는다.
    state : dict    직전 호출의 new_state. 최초 호출 시 {}.
    params : dict   config.yaml 의 params
    meta : dict     {'kind': 'price'|'rate', 'name': ..., 'ticker': ...}

    Returns
    -------
    decision : dict
    new_state : dict
    """
    meta = meta or {}
    row = df.iloc[-1]

    def g(col):
        v = row.get(col)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if f != f else f          # NaN 배제

    hist, rsi = g("osc_hist"), g("rsi")
    line, sl = g("osc_line"), g("osc_slope")
    bbw = g("bb_width")
    bbw_pctl = g("bb_width_pctile")

    # ---------- 1. 방향 ----------
    # 매수/매도 : 두 지표가 일치
    # 중립       : 두 지표가 엇갈림 (판정 보류 — 확인하지 못한 것을 확인한 척하지 않는다)
    # 관망       : 워밍업 중이라 아직 존이 정해지지 않음
    osc_zone = _zone(hist, state.get("osc_zone", "neutral"),
                     params["hist_upper"], params["hist_lower"])
    rsi_zone = _zone(rsi, state.get("rsi_zone", "neutral"),
                     params["rsi_upper"], params["rsi_lower"])

    prev_dir = state.get("direction", "관망")
    prev_conf = state.get("confirmed")          # 마지막으로 확정됐던 방향

    if osc_zone == "bull" and rsi_zone == "bull":
        direction, confirmed = "매수", "매수"
    elif osc_zone == "bear" and rsi_zone == "bear":
        direction, confirmed = "매도", "매도"
    elif "neutral" in (osc_zone, rsi_zone):
        direction, confirmed = "관망", prev_conf
    else:
        direction, confirmed = "중립", prev_conf   # 존이 엇갈림

    # ---------- 2. 변동성 경고 (매매에 개입하지 않음) ----------
    # 절대 임계치 대신 자산 자신의 최근 bb_width_window_weeks 분포 내 백분위로 판정한다
    # (2026-09: 종목별로 BB width 스케일이 너무 달라 절대 임계치 하나로는 형평이 안
    # 맞았다 — 자세한 이유는 core/indicators.py의 bb_width_percentile 참고).
    # 윈도우가 아직 안 찼으면(초기 구간) bbw_pctl 이 None → 경고 판정 보류.
    warn = bbw_pctl is not None and bbw_pctl >= params["bb_width_warn_percentile"]

    # 전환 = 확정 방향이 바뀐 것. 중립 진입/이탈은 전환으로 치지 않는다.
    changed = confirmed != prev_conf and confirmed is not None
    neutral_edge = (direction == "중립") != (prev_dir == "중립")

    # ---------- 3. 근거 ----------
    reason = (f"OSC {osc_zone}(hist {hist:+.3f})" if hist is not None else "OSC n/a")
    reason += f" · RSI {rsi_zone}({rsi:.1f})" if rsi is not None else " · RSI n/a"
    if direction == "중립":
        reason += f" · 지표 엇갈림, 직전 확정 {prev_conf or '없음'}"
    if changed:
        reason += f" · 확정전환 {prev_conf or '없음'}→{confirmed}"

    decision = {
        "direction": direction,
        "confirmed": confirmed,
        "changed": changed,
        "neutral_edge": neutral_edge,
        "reason": reason,
        "rsi": rsi,
        "osc_line": line,
        "osc_hist": hist,
        "osc_slope": sl,          # 참고용 원시값. 세기 판정에는 더 이상 쓰지 않는다.
        "bb_width": bbw,
        "bb_width_pctile": bbw_pctl,
        "flags": {"vol_warning": bool(warn)},
    }

    new_state = dict(state)               # 원본 in-place 수정 금지
    new_state.update({
        "osc_zone": osc_zone,
        "rsi_zone": rsi_zone,
        "direction": direction,
        "confirmed": confirmed,
        "last_date": str(df.index[-1].date()),
    })
    return decision, new_state
