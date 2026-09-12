"""
단일 self-contained HTML 생성. 외부 CDN/라이브러리 의존 없음.
데이터를 JSON으로 인라인 삽입하고 순수 SVG로 그린다.

패널 구성 (종목당 4단)
  1) 주봉 캔들 + 볼린저 밴드 + 확정 매수/매도 구간 배경 음영 (2026-09~)
  2) PPO / 시그널 / 히스토그램     (금리는 MACD, 단위 %p)
  3) RSI + 히스테리시스 밴드 (상/하한은 config.yaml의 rsi_upper/rsi_lower, 가변)
  4) BB width + 경고 마커 (최근 N주 분포 내 백분위 기준, 2026-09~)
"""
import json
import numpy as np
import pandas as pd
from .engine import run_signals

TPL = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>주봉 모니터 __DATE__</title><style>
:root{--bg:#0f1115;--fg:#e6e8eb;--dim:#8b929e;--grid:#232833;--up:#e2453c;--dn:#2f6fd0;
--l1:#f0b429;--l2:#7c8cf8;--warn:#ff6b6b;--ok:#4cd97b;--zbuy:#3ddc84;--zsell:#f5a623}
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
.card{margin:14px 16px;border:1px solid var(--grid);border-radius:10px;overflow:hidden;
scroll-margin-top:92px}
.hd{padding:10px 12px;background:#161a21;display:flex;justify-content:space-between;align-items:center}
.hd b{font-size:14px}.hd span{font-size:11px;color:var(--dim)}
svg{display:block;width:100%;height:auto}.rsn{padding:8px 12px;font-size:11px;color:var(--dim);
border-top:1px solid var(--grid)}
.tabs{display:none}.tabs.on{display:block}
/* 모바일에서 요약 테이블 오른쪽 컬럼(기울기~종가)이 카드 overflow:hidden 에
   잘려 아예 안 보이던 문제 — 테이블만 따로 가로 스크롤 가능한 래퍼로 감싼다. */
.twrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
/* 캔들 패널 아래 구간 확대(줌) 슬라이더 — 순수 CSS 듀얼 레인지 트릭.
   input 자체는 pointer-events:none 으로 뚫어놓고 썸(thumb)만 auto 로 살려서
   두 슬라이더가 겹쳐 있어도 각자의 손잡이만 잡아 끌 수 있게 한다. */
.zoomwrap{padding:2px 12px 10px}
.zlbl{font-size:10px;color:var(--dim);text-align:center;margin-bottom:4px}
.zsliders{position:relative;height:18px}
.zsliders input[type=range]{position:absolute;left:0;top:0;width:100%;margin:0;
-webkit-appearance:none;appearance:none;background:transparent;pointer-events:none}
.zsliders input[type=range]::-webkit-slider-runnable-track{height:4px;background:var(--grid);border-radius:2px}
.zsliders input[type=range]::-moz-range-track{height:4px;background:var(--grid);border-radius:2px}
.zsliders input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;
border-radius:50%;background:var(--fg);border:2px solid var(--bg);margin-top:-5px;pointer-events:auto;cursor:pointer}
.zsliders input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--fg);
border:2px solid var(--bg);pointer-events:auto;cursor:pointer}
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
  // 모바일 폭에서 종목/방향만 보이고 RSI~종가가 카드 밖으로 잘려 안 보이던 문제 →
  // 테이블을 가로 스크롤 가능한 .twrap 으로 감싼다(카드 자체의 overflow:hidden 은
  // 모서리 둥글게 처리용이라 그대로 두고, 스크롤은 안쪽 래퍼가 담당).
  // OSC 컬럼: 2026-09 이전엔 osc_line(PPO/MACD 원값)이 들어가 있어 실제 매수/매도
  // 판정에 쓰이는 히스토그램(osc_hist, generate_signal의 hist_upper/lower 비교 대상)과
  // 다른 값을 보여주고 있었다(예: S&P500 osc_line 3.658 vs 판정에 쓰인 osc_hist 0.190).
  // RSI 컬럼처럼 "판정에 실제로 쓰인 값 그대로"를 보여주도록 osc_hist로 교체.
  let h='<div class="card"><div class="twrap"><table><tr><th>종목</th><th>방향</th><th>RSI</th>'
   +'<th>OSC</th><th>기울기</th><th>BBW</th><th>종가</th></tr>';
  for(const r of rs){
    const cls=lblCls(r.direction), lbl=lblOf(r);
    h+=`<tr class="${r.changed?'chg':''}"><td><a href="#c_${r.id}" style="color:inherit">${esc(r.name)}</a>`
     +`${r.changed?' <span style="color:var(--l1)">◆</span>':(r.neutral_edge?' <span style="color:var(--dim)">·</span>':'')}</td>`
     +`<td class="${cls}">${lbl}</td><td>${f(r.rsi,1)}</td>`
     +`<td>${f(r.osc_hist,3)}</td><td>${f(r.osc_slope,4)}</td>`
     +`<td class="${r.vol_warning?'warn':''}">${f(r.bb_width,r.kind==='rate'?1:3)}${r.vol_warning?' ⚠':''}</td>`
     +`<td>${r.close.toLocaleString(undefined,{maximumFractionDigits:2})}</td></tr>`;
  }
  return h+'</table></div></div>';
}

const W=760,PAD=44;
function sc(v,lo,hi,y0,y1){if(hi===lo)return (y0+y1)/2;return y1-(v-lo)/(hi-lo)*(y1-y0);}
function axis(lo,hi,y0,y1,n=3){let s='';for(let i=0;i<=n;i++){const v=lo+(hi-lo)*i/n,y=sc(v,lo,hi,y0,y1);
 s+=`<line x1="${PAD}" y1="${y}" x2="${W-6}" y2="${y}" stroke="var(--grid)"/>`
  +`<text x="4" y="${y+3}" fill="var(--dim)" font-size="9">${Math.abs(v)>=1000?v.toFixed(0):v.toFixed(2)}</text>`;}
 return s;}
function ext(a){const v=a.filter(x=>x!=null&&isFinite(x));if(!v.length)return[0,1];
 let lo=Math.min(...v),hi=Math.max(...v);const p=(hi-lo)*0.08||1;return[lo-p,hi+p];}

// x축(연/월) 눈금 — 6개월(1월/7월) 간격으로만 표시한다(2026-09~, 기존엔 라벨이
// 안 겹치게 최대 9개로 적당히 솎아내는 방식이었는데 간격이 들쭉날쭉했음).
function sixMonthTicks(t){
  const idx=[];
  for(let i=0;i<t.length;i++){
    const ym=t[i].slice(0,7);                       // "YYYY-MM"
    if(i>0 && ym===t[i-1].slice(0,7)) continue;      // 같은 달 중복 제거
    const mm=+ym.slice(5,7);
    if(mm===1||mm===7) idx.push(i);
  }
  return idx;
}

// body 는 이미 절대 y 좌표로 그려지므로 transform 을 쓰면 이중 오프셋이 된다.
function panel(s,H,body,label){
 return `<text x="4" y="${s+10}" fill="var(--dim)" font-size="9">${label}</text>${body}`;}

// ixLo/ixHi: r.series 전체 중 실제로 그릴 구간의 인덱스 범위(포함) — 하단 확대
// 슬라이더가 이 범위를 좁히면 x축 간격과 y축 스케일(ext())이 그 구간 값만
// 기준으로 다시 계산되어 자동으로 확대/재조정된다. 기본은 전체 범위(0~n-1).
function chartSVG(r, ixLo, ixHi){
  const s0=r.series, sl=a=>a.slice(ixLo,ixHi+1);
  const d={t:sl(s0.t),o:sl(s0.o),h:sl(s0.h),l:sl(s0.l),c:sl(s0.c),
    bu:sl(s0.bu),bm:sl(s0.bm),bl:sl(s0.bl),ol:sl(s0.ol),os:sl(s0.os),oh:sl(s0.oh),
    r:sl(s0.r),w:sl(s0.w),wp:sl(s0.wp),cf:s0.cf?sl(s0.cf):null};
  const n=d.t.length,x=i=>PAD+(W-PAD-8)*(n<2?0.5:i/(n-1)),bw=Math.max(1.2,(W-PAD-8)/n*0.6);
  const H1=190,H2=95,H3=80,H4=70,G=16;let y=0,out='';
  const TOP=20;   // 패널 제목(y+10)과 y축 맨 위 눈금값이 너무 붙어 보여 여유를 더 둠(기존 14)

  // 1) 캔들 + BB (+ 확정 매수/매도 구간 배경 음영, 2026-09~)
  // d.cf[i] = 그 주 시점의 confirmed(마지막 확정 방향). run_signals 순차 재실행으로
  // 얻은 이력이라 라이브 판정(state.json 이어가기)과 동일하게 재현된다 —
  // core/chart.py의 _confirmed_series 참고. 워밍업 이전 구간은 null(무색).
  // 음영색은 캔들(빨강=상승봉/파랑=하락봉)과 겹쳐 헷갈리지 않도록 별도 배색
  // (초록=매수, 주황=매도, --zbuy/--zsell)을 쓰고, 처음엔 opacity 0.10으로 너무
  // 옅어 안 보인다는 제보(2026-09)가 있어 0.24로 올림. 배경(bg)은 제목·범례·축·
  // 캔들 등 다른 모든 요소보다 먼저(=맨 뒤에) 그려서 확실히 뒤에 깔리게 한다
  // (2026-09 제보 — 그래야 캔들이 항상 배경 위에서 또렷하게 보인다).
  // 금리(kind=rate) 자산은 방향 라벨을 매수/매도 대신 상승/하락으로 쓰므로
  // (요약표 lblOf/RM과 동일 관례) 범례 텍스트도 맞춰서 바꾼다.
  const [lo,hi]=ext([...d.h,...d.l,...d.bu,...d.bl]);
  const slot=(W-PAD-8)/n;
  const zLbl=r.kind==='rate'?{buy:'상승구간',sell:'하락구간'}:{buy:'매수구간',sell:'매도구간'};
  let bg='';
  if(d.cf){
    let j=0;
    while(j<n){
      const v=d.cf[j];
      if(v==null){j++;continue;}
      let k=j;while(k+1<n&&d.cf[k+1]===v)k++;
      const col=v==='매수'?'var(--zbuy)':(v==='매도'?'var(--zsell)':null);
      // 배경 상단을 패널 맨 위(y)가 아니라 TOP만큼 내려서 시작 — 그 위 제목("주봉 +
      // 볼린저...")과 매수/매도구간 범례 글자를 음영이 가리던 문제(2026-09 제보) 수정.
      if(col) bg+=`<rect x="${x(j)-slot/2}" y="${y+TOP}" width="${x(k)-x(j)+slot}" height="${H1-TOP}" fill="${col}" opacity="0.24"/>`;
      j=k+1;
    }
  }
  let s=bg;   // ← 배경이 항상 이 패널의 첫 번째(=맨 뒤) 요소가 된다.
  s+=`<text x="4" y="${y+10}" fill="var(--dim)" font-size="9">주봉 + 볼린저(${D.p.bb_period},${D.p.bb_std})</text>`;
  s+=`<rect x="${W-124}" y="${y+3}" width="7" height="7" fill="var(--zbuy)" opacity="0.9"/>`
   +`<text x="${W-113}" y="${y+10}" fill="var(--dim)" font-size="9">${zLbl.buy}</text>`
   +`<rect x="${W-60}" y="${y+3}" width="7" height="7" fill="var(--zsell)" opacity="0.9"/>`
   +`<text x="${W-49}" y="${y+10}" fill="var(--dim)" font-size="9">${zLbl.sell}</text>`;
  s+=axis(lo,hi,y+TOP,y+H1);
  for(let i=0;i<n;i++){if(d.bu[i]==null)continue;
    s+=`<circle cx="${x(i)}" cy="${sc(d.bu[i],lo,hi,y+TOP,y+H1)}" r="0.7" fill="var(--dim)"/>`
      +`<circle cx="${x(i)}" cy="${sc(d.bl[i],lo,hi,y+TOP,y+H1)}" r="0.7" fill="var(--dim)"/>`
      +`<circle cx="${x(i)}" cy="${sc(d.bm[i],lo,hi,y+TOP,y+H1)}" r="0.5" fill="#4a5163"/>`;}
  for(let i=0;i<n;i++){const up=d.c[i]>=d.o[i],col=up?'var(--up)':'var(--dn)';
    const yh=sc(d.h[i],lo,hi,y+TOP,y+H1),yl=sc(d.l[i],lo,hi,y+TOP,y+H1);
    const yo=sc(d.o[i],lo,hi,y+TOP,y+H1),yc=sc(d.c[i],lo,hi,y+TOP,y+H1);
    s+=`<line x1="${x(i)}" y1="${yh}" x2="${x(i)}" y2="${yl}" stroke="${col}" stroke-width="0.8"/>`
      +`<rect x="${x(i)-bw/2}" y="${Math.min(yo,yc)}" width="${bw}" height="${Math.max(1,Math.abs(yc-yo))}" fill="${col}"/>`;}
  out+=s;y+=H1+G;

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

  // x축 연/월 라벨 — 6개월(1월/7월) 간격 텍스트만 표시한다. 이전엔 4개 패널을
  // 관통하는 점선 세로 눈금도 같이 그렸는데, 캔들/음영 위로 선이 겹쳐 번잡하다는
  // 제보(2026-09)로 세로선은 없애고 텍스트 라벨만 남김.
  for(const i of sixMonthTicks(d.t)){
    out+=`<text x="${x(i)}" y="${y+11}" fill="var(--dim)" font-size="9" text-anchor="middle">${d.t[i].slice(2,7).replace('-','.')}</text>`;
  }
  y+=16;

  return `<svg viewBox="0 0 ${W} ${y}">${out}</svg>`;
}

// 캔들 패널 아래 구간 확대 슬라이더 — 좌/우 손잡이 사이 구간만 chartSVG로
// 다시 그린다(같은 함수가 y축 범위를 그 구간 값만으로 재계산하므로 자동 확대/
// 축소가 됨). 최소 구간(minSpan)은 너무 좁혀서 축이 무의미해지는 것을 막는다.
function chart(r){
  const n0=r.series.t.length;
  const lbl=lblOf(r), cls=lblCls(r.direction);
  return `<div class="card" id="c_${r.id}"><div class="hd"><b>${esc(r.name)}</b>`
   +`<span class="${cls}">${lbl}</span></div>`
   +`<div class="svgholder" id="sv_${r.id}">${chartSVG(r,0,n0-1)}</div>`
   +`<div class="zoomwrap"><div class="zlbl" id="zl_${r.id}">${r.series.t[0]} ~ ${r.series.t[n0-1]} (전체 ${n0}주 · 드래그로 확대)</div>`
   +`<div class="zsliders">`
   +`<input type="range" class="zl" data-id="${r.id}" min="0" max="${n0-1}" step="1" value="0">`
   +`<input type="range" class="zr" data-id="${r.id}" min="0" max="${n0-1}" step="1" value="${n0-1}">`
   +`</div></div>`
   +`<div class="rsn" id="rs_${r.id}">${esc(r.reason)}  ·  ${r.series.t[0]} ~ ${r.series.t[n0-1]}  ·  ${n0}주</div></div>`;
}

// 슬라이더는 카드마다 새로 만들어지므로(탭 전환 시 #root 통째로 다시 렌더링)
// 개별 리스너 대신 #root 에 한 번만 위임 리스너를 걸어둔다.
R.addEventListener('input', e=>{
  const t=e.target;
  if(!(t.classList.contains('zl')||t.classList.contains('zr'))) return;
  const id=t.dataset.id, row=D.rows[id];
  const wrap=t.closest('.zsliders');
  const zl=wrap.querySelector('.zl'), zr=wrap.querySelector('.zr');
  let a=+zl.value,b=+zr.value;
  if(a>b){const tmp=a;a=b;b=tmp;}
  const n0=row.series.t.length,minSpan=Math.min(8,n0-1);
  if(b-a<minSpan){
    if(t===zl) a=Math.max(0,b-minSpan); else b=Math.min(n0-1,a+minSpan);
    zl.value=a;zr.value=b;
  }
  document.getElementById('sv_'+id).innerHTML=chartSVG(row,a,b);
  document.getElementById('zl_'+id).textContent=`${row.series.t[a]} ~ ${row.series.t[b]} (전체 ${n0}주 · 드래그로 확대)`;
  document.getElementById('rs_'+id).textContent=`${row.reason}  ·  ${row.series.t[a]} ~ ${row.series.t[b]}  ·  ${b-a+1}주`;
});

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


def _confirmed_history(df, params, meta):
    """
    자산의 전체 주봉 히스토리에 run_signals(core/engine.py, 백테스트 순차 재실행)를
    돌려 매주 confirmed(마지막 확정 매수/매도) 이력을 구한다. generate_signal은
    순수 함수이고 run_signals는 매주 df.iloc[:i+1]만 순차로 넘기므로, 이 결과는
    run_weekly.py가 state.json으로 이어가는 라이브 판정과 동일하게 재현된다
    (콜드스타트 로직이 이미 이 성질에 기대고 있음, run_weekly.py 참고).
    워밍업(slow+signal 주) 이전 구간은 결과에 없어 조회 시 None이 된다.
    """
    hist = run_signals(df, params, meta)
    return hist["confirmed"] if not hist.empty else pd.Series(dtype=object)


def build_html(payload: list, params: dict, asof: str, tail: int = 260) -> str:
    rows = []
    for i, (e, df, dec) in enumerate(payload):
        d = df.tail(tail)
        conf_hist = _confirmed_history(df, params, {"kind": e["kind"], **e})
        cf = [conf_hist.get(ts) for ts in d.index]
        rows.append({
            "id": i, "name": e["name"], "group": e["group"], "kind": e["kind"],
            "direction": dec["direction"], "confirmed": dec["confirmed"],
            "changed": bool(dec["changed"]),
            "neutral_edge": bool(dec["neutral_edge"]), "reason": dec["reason"],
            "rsi": dec["rsi"], "osc_line": dec["osc_line"], "osc_hist": dec["osc_hist"],
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
                "cf": cf,
            },
        })
    data = json.dumps({"rows": rows, "p": params}, ensure_ascii=False)
    h = TPL.replace("__DATA__", data).replace("__JS__", JS).replace("__DATE__", asof)
    for k, v in [("__F__", "fast"), ("__S__", "slow"), ("__G__", "signal"),
                 ("__R__", "rsi_period"), ("__RL__", "rsi_lower"), ("__RU__", "rsi_upper")]:
        h = h.replace(k, str(params[v]))
    return h
