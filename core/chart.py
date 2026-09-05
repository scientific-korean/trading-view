"""
단일 self-contained HTML 생성. 외부 CDN/라이브러리 의존 없음.
데이터를 JSON으로 인라인 삽입하고 순수 SVG로 그린다.

패널 구성 (종목당 4단)
  1) 주봉 캔들 + 볼린저 밴드
  2) PPO / 시그널 / 히스토그램     (금리는 MACD, 단위 %p)
  3) RSI + 히스테리시스 밴드 (상/하한은 config.yaml의 rsi_upper/rsi_lower, 가변)
  4) BB width + 경고 마커 (최근 N주 분포 내 백분위 기준, 2026-09~)
"""
import json
import numpy as np
import pandas as pd

TPL = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>주봉 모니터 __DATE__</title><style>
:root{--bg:#0f1115;--fg:#e6e8eb;--dim:#8b929e;--grid:#232833;--up:#e2453c;--dn:#2f6fd0;
--l1:#f0b429;--l2:#7c8cf8;--warn:#ff6b6b;--ok:#4cd97b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",sans-serif}
header{padding:16px;border-bottom:1px solid var(--grid);position:sticky;top:0;background:var(--bg);z-index:9}
h1{margin:0;font-size:17px}.sub{color:var(--dim);font-size:12px;margin-top:4px}
#nav{display:flex;gap:6px;overflow-x:auto;padding:10px 16px;border-bottom:1px solid var(--grid)}
#nav button{background:#1a1e26;color:var(--dim);border:1px solid var(--grid);border-radius:14px;
padding:5px 11px;font-size:12px;white-space:nowrap;cursor:pointer}
#nav button.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--dim);font-weight:500;font-size:11px}td:first-child,th:first-child{text-align:left}
.buy{color:var(--up);font-weight:600}.sell{color:var(--dn);font-weight:600}
.hold{color:var(--dim)}.neu{color:var(--l1)}
.chg{background:#2a1f14}.warn{color:var(--warn)}
.card{margin:14px 16px;border:1px solid var(--grid);border-radius:10px;overflow:hidden}
.hd{padding:10px 12px;background:#161a21;display:flex;justify-content:space-between;align-items:center}
.hd b{font-size:14px}.hd span{font-size:11px;color:var(--dim)}
svg{display:block;width:100%;height:auto}.rsn{padding:8px 12px;font-size:11px;color:var(--dim);
border-top:1px solid var(--grid)}
.tabs{display:none}.tabs.on{display:block}
</style></head><body>
<header><h1>주봉 신호 모니터</h1><div class="sub">기준일 __DATE__ · PPO(__F__,__S__,__G__) · RSI(__R__) 존 __RL__/__RU__ · AND 대칭</div></header>
<div id="nav"></div><div id="root"></div>
<script>const D=__DATA__;</script>
<script>__JS__</script></body></html>"""

JS = r"""
const R=document.getElementById('root'),N=document.getElementById('nav');
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const f=(v,d=2)=>v==null||!isFinite(v)?'—':v.toFixed(d);
const groups=[...new Set(D.rows.map(r=>r.group))];
const RM={'매수':'상승','매도':'하락'};
const lblCls=d=>d==='매수'?'buy':d==='매도'?'sell':d==='중립'?'neu':'hold';
function lblOf(r){
  const m=x=>r.kind==='rate'?(RM[x]||x):x;
  if(r.direction==='중립') return r.confirmed?`중립(직전 ${m(r.confirmed)})`:'중립';
  return m(r.direction);
}

function summary(g){
  const rs=D.rows.filter(r=>g==='전체'||r.group===g);
  let h='<div class="card"><table><tr><th>종목</th><th>방향</th><th>RSI</th>'
   +'<th>OSC</th><th>기울기</th><th>BBW</th><th>종가</th></tr>';
  for(const r of rs){
    const cls=lblCls(r.direction), lbl=lblOf(r);
    h+=`<tr class="${r.changed?'chg':''}"><td><a href="#c_${r.id}" style="color:inherit">${esc(r.name)}</a>`
     +`${r.changed?' <span style="color:var(--l1)">◆</span>':(r.neutral_edge?' <span style="color:var(--dim)">·</span>':'')}</td>`
     +`<td class="${cls}">${lbl}</td><td>${f(r.rsi,1)}</td>`
     +`<td>${f(r.osc_line,3)}</td><td>${f(r.osc_slope,4)}</td>`
     +`<td class="${r.vol_warning?'warn':''}">${f(r.bb_width,r.kind==='rate'?1:3)}${r.vol_warning?' ⚠':''}</td>`
     +`<td>${r.close.toLocaleString(undefined,{maximumFractionDigits:2})}</td></tr>`;
  }
  return h+'</table></div>';
}

const W=760,PAD=44;
function sc(v,lo,hi,y0,y1){if(hi===lo)return (y0+y1)/2;return y1-(v-lo)/(hi-lo)*(y1-y0);}
function axis(lo,hi,y0,y1,n=3){let s='';for(let i=0;i<=n;i++){const v=lo+(hi-lo)*i/n,y=sc(v,lo,hi,y0,y1);
 s+=`<line x1="${PAD}" y1="${y}" x2="${W-6}" y2="${y}" stroke="var(--grid)"/>`
  +`<text x="4" y="${y+3}" fill="var(--dim)" font-size="9">${Math.abs(v)>=1000?v.toFixed(0):v.toFixed(2)}</text>`;}
 return s;}
function ext(a){const v=a.filter(x=>x!=null&&isFinite(x));if(!v.length)return[0,1];
 let lo=Math.min(...v),hi=Math.max(...v);const p=(hi-lo)*0.08||1;return[lo-p,hi+p];}

// x축(연/월) 눈금 — 월이 바뀌는 지점을 찾은 뒤 라벨이 안 겹치게 적당히 솎아낸다.
function monthTicks(t, maxLabels=9){
  const idx=[];
  for(let i=0;i<t.length;i++){
    const ym=t[i].slice(0,7);           // "YYYY-MM"
    if(i===0||ym!==t[i-1].slice(0,7)) idx.push(i);
  }
  if(idx.length<=maxLabels) return idx;
  const step=Math.ceil(idx.length/maxLabels);
  return idx.filter((_,k)=>k%step===0);
}

// body 는 이미 절대 y 좌표로 그려지므로 transform 을 쓰면 이중 오프셋이 된다.
function panel(s,H,body,label){
 return `<text x="4" y="${s+10}" fill="var(--dim)" font-size="9">${label}</text>${body}`;}

function chart(r){
  const d=r.series,n=d.t.length,x=i=>PAD+(W-PAD-8)*(n<2?0.5:i/(n-1)),bw=Math.max(1.2,(W-PAD-8)/n*0.6);
  const H1=190,H2=95,H3=80,H4=70,G=16;let y=0,out='';
  const TOP=20;   // 패널 제목(y+10)과 y축 맨 위 눈금값이 너무 붙어 보여 여유를 더 둠(기존 14)

  // 1) 캔들 + BB
  const [lo,hi]=ext([...d.h,...d.l,...d.bu,...d.bl]);
  let s=axis(lo,hi,y+TOP,y+H1);
  for(let i=0;i<n;i++){if(d.bu[i]==null)continue;
    s+=`<circle cx="${x(i)}" cy="${sc(d.bu[i],lo,hi,y+TOP,y+H1)}" r="0.7" fill="var(--dim)"/>`
      +`<circle cx="${x(i)}" cy="${sc(d.bl[i],lo,hi,y+TOP,y+H1)}" r="0.7" fill="var(--dim)"/>`
      +`<circle cx="${x(i)}" cy="${sc(d.bm[i],lo,hi,y+TOP,y+H1)}" r="0.5" fill="#4a5163"/>`;}
  for(let i=0;i<n;i++){const up=d.c[i]>=d.o[i],col=up?'var(--up)':'var(--dn)';
    const yh=sc(d.h[i],lo,hi,y+TOP,y+H1),yl=sc(d.l[i],lo,hi,y+TOP,y+H1);
    const yo=sc(d.o[i],lo,hi,y+TOP,y+H1),yc=sc(d.c[i],lo,hi,y+TOP,y+H1);
    s+=`<line x1="${x(i)}" y1="${yh}" x2="${x(i)}" y2="${yl}" stroke="${col}" stroke-width="0.8"/>`
      +`<rect x="${x(i)-bw/2}" y="${Math.min(yo,yc)}" width="${bw}" height="${Math.max(1,Math.abs(yc-yo))}" fill="${col}"/>`;}
  out+=panel(y,H1,s,'주봉 + 볼린저('+D.p.bb_period+','+D.p.bb_std+')');y+=H1+G;

  // 2) PPO / MACD
  const [l2,h2]=ext([...d.ol,...d.os,...d.oh]);
  s=axis(l2,h2,y+TOP,y+H2);
  const z=sc(0,l2,h2,y+TOP,y+H2);
  s+=`<line x1="${PAD}" y1="${z}" x2="${W-6}" y2="${z}" stroke="#4a5163" stroke-dasharray="2,2"/>`;
  for(let i=0;i<n;i++){if(d.oh[i]==null)continue;const yy=sc(d.oh[i],l2,h2,y+TOP,y+H2);
    s+=`<rect x="${x(i)-bw/2}" y="${Math.min(yy,z)}" width="${bw}" height="${Math.max(0.6,Math.abs(yy-z))}" fill="${d.oh[i]>=0?'var(--up)':'var(--dn)'}" opacity="0.55"/>`;}
  const path=(a,c)=>{let p='',st=true;for(let i=0;i<n;i++){if(a[i]==null||!isFinite(a[i])){st=true;continue;}
    p+=(st?'M':'L')+x(i)+' '+sc(a[i],l2,h2,y+TOP,y+H2);st=false;}
    return `<path d="${p}" fill="none" stroke="${c}" stroke-width="1.3"/>`;};
  s+=path(d.ol,'var(--l1)')+path(d.os,'var(--l2)');
  out+=panel(y,H2,s,(r.kind==='rate'?'MACD(%p) ':'PPO(%) ')+D.p.fast+','+D.p.slow+','+D.p.signal);y+=H2+G;

  // 3) RSI
  s=axis(0,100,y+TOP,y+H3,2);
  const yU=sc(D.p.rsi_upper,0,100,y+TOP,y+H3),yL=sc(D.p.rsi_lower,0,100,y+TOP,y+H3);
  s+=`<rect x="${PAD}" y="${yU}" width="${W-PAD-6}" height="${yL-yU}" fill="#ffffff" opacity="0.05"/>`
   +`<line x1="${PAD}" y1="${yU}" x2="${W-6}" y2="${yU}" stroke="var(--ok)" stroke-dasharray="3,3" opacity=".6"/>`
   +`<line x1="${PAD}" y1="${yL}" x2="${W-6}" y2="${yL}" stroke="var(--warn)" stroke-dasharray="3,3" opacity=".6"/>`;
  let p='',st=true;for(let i=0;i<n;i++){if(d.r[i]==null){st=true;continue;}
    p+=(st?'M':'L')+x(i)+' '+sc(d.r[i],0,100,y+TOP,y+H3);st=false;}
  s+=`<path d="${p}" fill="none" stroke="var(--l1)" stroke-width="1.3"/>`;
  out+=panel(y,H3,s,'RSI('+D.p.rsi_period+') 히스테리시스 '+D.p.rsi_lower+'/'+D.p.rsi_upper);y+=H3+G;

  // 4) BB width — 절대 임계치가 아니라 최근 bb_width_window_weeks 분포 내 백분위로
  // 경고를 판정하므로(자산마다 스케일이 달라 고정선 하나로는 형평이 안 맞음),
  // 고정 수평선 대신 실제로 경고가 뜬 주(週)를 점으로 표시한다.
  const [l4,h4]=ext([...d.w]);s=axis(l4,h4,y+TOP,y+H4,2);
  p='';st=true;for(let i=0;i<n;i++){if(d.w[i]==null){st=true;continue;}
    p+=(st?'M':'L')+x(i)+' '+sc(d.w[i],l4,h4,y+TOP,y+H4);st=false;}
  s+=`<path d="${p}" fill="none" stroke="#9aa4b8" stroke-width="1.2"/>`;
  for(let i=0;i<n;i++){if(d.w[i]==null||d.wp[i]==null||d.wp[i]<D.p.bb_width_warn_percentile)continue;
    s+=`<circle cx="${x(i)}" cy="${sc(d.w[i],l4,h4,y+TOP,y+H4)}" r="2.2" fill="var(--warn)"/>`;}
  // "BB width" 패널 제목과 같은 높이(y+10)에 둬서 아래 실제 데이터 선과 겹치지 않게 함.
  s+=`<text x="${W-8}" y="${y+10}" fill="var(--warn)" font-size="9" text-anchor="end">`
   +`⚠ 최근 ${D.p.bb_width_window_weeks}주 상위 ${100-D.p.bb_width_warn_percentile}%</text>`;
  out+=panel(y,H4,s,'BB width'+(r.kind==='rate'?' (bp)':''));y+=H4+10;

  // x축 연/월 라벨 — 4개 패널을 관통하는 점선 눈금 + 맨 아래 "YY.MM" 텍스트
  const chartBottom=y;
  for(const i of monthTicks(d.t)){
    const xx=x(i);
    out+=`<line x1="${xx}" y1="0" x2="${xx}" y2="${chartBottom}" stroke="var(--grid)" stroke-dasharray="1,3" opacity="0.5"/>`
      +`<text x="${xx}" y="${chartBottom+11}" fill="var(--dim)" font-size="9" text-anchor="middle">${d.t[i].slice(2,7).replace('-','.')}</text>`;
  }
  y+=16;

  const lbl=lblOf(r), cls=lblCls(r.direction);
  return `<div class="card" id="c_${r.id}"><div class="hd"><b>${esc(r.name)}</b>`
   +`<span class="${cls}">${lbl}</span></div>`
   +`<svg viewBox="0 0 ${W} ${y}">${out}</svg>`
   +`<div class="rsn">${esc(r.reason)}  ·  ${d.t[0]} ~ ${d.t[n-1]}  ·  ${n}주</div></div>`;
}

function render(g){
  R.innerHTML=summary(g)+D.rows.filter(r=>g==='전체'||r.group===g).map(chart).join('');
  [...N.children].forEach(b=>b.classList.toggle('on',b.textContent===g));
}
['전체',...groups].forEach(g=>{const b=document.createElement('button');b.textContent=g;
  b.onclick=()=>render(g);N.appendChild(b);});
render('전체');
"""


def _ser(s, nd=6):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), nd) for v in s]


def build_html(payload: list, params: dict, asof: str, tail: int = 260) -> str:
    rows = []
    for i, (e, df, dec) in enumerate(payload):
        d = df.tail(tail)
        rows.append({
            "id": i, "name": e["name"], "group": e["group"], "kind": e["kind"],
            "direction": dec["direction"], "confirmed": dec["confirmed"],
            "changed": bool(dec["changed"]),
            "neutral_edge": bool(dec["neutral_edge"]), "reason": dec["reason"],
            "rsi": dec["rsi"], "osc_line": dec["osc_line"],
            "osc_slope": dec["osc_slope"], "bb_width": dec["bb_width"],
            "vol_warning": dec["flags"]["vol_warning"],
            "close": float(df["close"].iloc[-1]),
            "series": {
                "t": [str(x.date()) for x in d.index],
                "o": _ser(d["open"]), "h": _ser(d["high"]),
                "l": _ser(d["low"]), "c": _ser(d["close"]),
                "bu": _ser(d["bb_upper"]), "bm": _ser(d["bb_mid"]), "bl": _ser(d["bb_lower"]),
                "ol": _ser(d["osc_line"]), "os": _ser(d["osc_signal"]), "oh": _ser(d["osc_hist"]),
                "r": _ser(d["rsi"]), "w": _ser(d["bb_width"]), "wp": _ser(d["bb_width_pctile"]),
            },
        })
    data = json.dumps({"rows": rows, "p": params}, ensure_ascii=False)
    h = TPL.replace("__DATA__", data).replace("__JS__", JS).replace("__DATE__", asof)
    for k, v in [("__F__", "fast"), ("__S__", "slow"), ("__G__", "signal"),
                 ("__R__", "rsi_period"), ("__RL__", "rsi_lower"), ("__RU__", "rsi_upper")]:
        h = h.replace(k, str(params[v]))
    return h
