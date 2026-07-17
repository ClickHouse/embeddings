#!/usr/bin/env python3
# Build results/recall.json + site/recall.html (self-contained heatmap viewer) from results/recall.csv
import csv, json, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for r in csv.DictReader(open(os.path.join(BASE, 'results/recall.csv'))):
    if not r['recall100']:
        continue
    rows.append({'ds': r['dataset'].replace('emb_', ''), 'ty': r['type'], 'ro': r['rotation'],
                 'b': int(r['bits']), 'd': int(r['dims']),
                 'recall10': float(r['recall10']), 'recall100': float(r['recall100']),
                 'recall10in100': float(r['recall10in100'])})
data = json.dumps(rows, separators=(',', ':'))
json.dump(rows, open(os.path.join(BASE, 'results/recall.json'), 'w'), separators=(',', ':'))

HTML = r'''<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>QBit recall heatmap</title>
<style>
  body { background:#111; color:#ddd; font:16px system-ui,-apple-system,sans-serif; margin:0; padding:1em 1.2em; }
  h1 { font-size:19px; color:#ffd54f; margin:.2em 0 .1em; }
  .sub { color:#888; font-size:14px; margin-bottom:.6em; }
  .ctl { display:flex; flex-wrap:wrap; gap:.35em 1.4em; align-items:center; margin:.5em 0 1em; }
  .ctl > div { display:flex; align-items:center; gap:.3em; }
  .ctl label { color:#999; font-size:13px; text-transform:uppercase; letter-spacing:.04em; }
  .opt { padding:.16em .55em; background:#2b2b2b; border:1px solid #000; cursor:pointer; user-select:none; font-size:14px; border-radius:3px; }
  .opt:hover { background:#ffd54f; color:#000; }
  .opt.sel { background:#fff; color:#000; }
  table { border-collapse:collapse; }
  caption { caption-side:top; text-align:left; color:#8bc34a; font-family:monospace; margin-bottom:.45em; }
  th, td { padding:.28em .5em; text-align:center; font-variant-numeric:tabular-nums; font-size:14px; }
  th { color:#9a9a9a; font-weight:normal; }
  th.cor { color:#ffd54f; }
  td.cell { color:#000; font-weight:600; min-width:44px; border:1px solid #111; transition:opacity .08s; line-height:1.15; }
  td.cell.dim { opacity:.15; }         /* on hover: cells below the pointed value fade, so >= cells stand out */
  td.cell .sz { display:block; font-size:11px; font-weight:400; opacity:.6; }   /* bytes/vector, second line */
  td.cell.pareto { font-weight:900; }   /* pareto frontier (best recall per byte budget) — bold only */
  td.cell.pareto .sz { opacity:.85; }
  /* among the highlighted (>= threshold) cells, the single best recall-per-byte — bright outline + glow */
  td.cell.best { outline:2px solid #fff; outline-offset:-2px; box-shadow:0 0 12px 3px rgba(255,255,255,.9);
                 filter:brightness(1.18); position:relative; z-index:2; }
  .legend { display:flex; align-items:center; gap:.4em; margin-top:1em; color:#999; font-size:13px; }
  .bar { width:220px; height:14px; border-radius:2px; cursor:crosshair;
         background:linear-gradient(90deg,hsl(0,75%,50%),hsl(60,75%,50%),hsl(120,75%,50%)); }
  .legval { color:#ffd54f; font-family:monospace; min-width:4em; }
  .chartwrap { margin-top:1.6em; }
  .csub { color:#8bc34a; font-family:monospace; font-size:13px; margin-bottom:.35em; }
  svg.chart { width:100%; max-width:720px; height:auto; background:#181818; border-radius:4px; }
  svg.chart .grid { stroke:#333; stroke-width:1; }
  svg.chart .ax   { fill:#888; font-size:11px; font-family:monospace; }
  svg.chart .axt  { fill:#aaa; font-size:12px; }
  svg.chart .front{ fill:none; stroke:#ffd54f; stroke-width:2; }
</style></head><body>
<h1>QBit representation recall &mdash; bits &times; dims heatmap</h1>
<div class=sub>100k-row sample per dataset, 20 random queries, exact-cosine ground truth. Green = 1.0, red = 0.0.</div>
<div class=ctl>
  <div><label>dataset</label><span id=sw-ds></span></div>
  <div><label>type</label><span id=sw-ty></span></div>
  <div><label>rotation</label><span id=sw-ro></span></div>
  <div><label>metric</label><span id=sw-me></span></div>
</div>
<div id=tbl></div>
<div class=legend>0.0 <span class=bar id=bar></span> 1.0 <span id=legval class=legval></span></div>
<div class=chartwrap>
  <div class=csub>best achievable recall vs. size &mdash; line = pareto frontier, dots = all configs (color = recall)</div>
  <div id=chart></div>
</div>
<script>
const DATA = __DATA__;
const METRICS = { recall100:'Recall@100', recall10:'Recall@10', recall10in100:'Recall 10-in-100' };
let st = { ds:'siglip2', ty:'Int8', ro:'rotated', me:'recall100' };
const uniq = k => [...new Set(DATA.map(r => r[k]))];
const color = v => v==null ? '#222' : `hsl(${(v*120).toFixed(0)},75%,50%)`;
const fmtB = n => n < 1024 ? n+'B' : (n/1024).toFixed(1).replace(/\.0$/,'')+'K';
const bytesPerVec = (b,d) => fmtB(b*d/8);
function buildSwitch(el, keys, cur, onPick, labels) {
  el.innerHTML = '';
  keys.forEach(k => {
    const s = document.createElement('span');
    s.className = 'opt' + (k===cur ? ' sel' : '');
    s.textContent = labels ? labels[k] : k;
    s.onclick = () => { [...el.children].forEach(c => c.classList.remove('sel')); s.classList.add('sel'); onPick(k); };
    el.appendChild(s);
  });
}
function render() {
  const rows = DATA.filter(r => r.ds===st.ds && r.ty===st.ty && r.ro===st.ro);
  const bits = [...new Set(rows.map(r => r.b))].sort((a,b)=>a-b);
  const dims = [...new Set(rows.map(r => r.d))].sort((a,b)=>a-b);
  const m = {}; rows.forEach(r => m[r.b + '_' + r.d] = r[st.me]);
  // pareto frontier: a cell is on it if no OTHER cell with fewer bytes has a higher recall
  const cells = rows.map(r => ({k:r.b+'_'+r.d, v:r[st.me], bytes:r.b*r.d/8})).sort((a,b)=>a.bytes-b.bytes);
  const pareto = new Set(); let runMax = -Infinity, ci = 0;
  while (ci < cells.length) {
    let j = ci; const by = cells[ci].bytes;
    while (j < cells.length && cells[j].bytes === by) j++;                    // group of equal-bytes cells [ci,j)
    for (let k=ci; k<j; k++) if (cells[k].v >= runMax) pareto.add(cells[k].k);  // no strictly-cheaper cell beats its recall
    for (let k=ci; k<j; k++) if (cells[k].v > runMax) runMax = cells[k].v;
    ci = j;
  }
  let h = `<table><caption>${st.ds} &middot; ${st.ty} &middot; ${st.ro} &middot; ${METRICS[st.me]}</caption>`;
  h += '<tr><th class=cor>bits&nbsp;\\&nbsp;dims</th>' + dims.map(d => `<th>${d}</th>`).join('') + '</tr>';
  for (const b of bits) {
    h += `<tr><th class=cor>${b}</th>` + dims.map(d => {
      const v = m[b + '_' + d];
      return v==null ? '<td></td>' : `<td class="cell${pareto.has(b+'_'+d)?' pareto':''}" data-v="${v}" data-bytes="${b*d/8}" style="background:${color(v)}" title="bits=${b} dims=${d}">${v.toFixed(2)}<span class=sz>${bytesPerVec(b,d)}</span></td>`;
    }).join('') + '</tr>';
  }
  h += '</table>';
  document.getElementById('tbl').innerHTML = h;

  // ---- pareto chart: best recall (y) vs bytes/vector (x, log scale). NB: all SVG attrs MUST be
  //      quoted — an unquoted value before a self-closing "/>" swallows the slash (y2="316/") ----
  const P = rows.map(r => ({b:r.b, d:r.d, bytes:r.b*r.d/8, v:r[st.me], pa:pareto.has(r.b+'_'+r.d)}));
  const CW=680, CH=360, ML=54, MR=18, MT=16, MB=44, IW=CW-ML-MR, IH=CH-MT-MB;
  const bs = P.map(p=>p.bytes), lo=Math.log2(Math.min(...bs)), hi=Math.log2(Math.max(...bs)), spanx=(hi-lo)||1;
  const X = by => (ML + (Math.log2(by)-lo)/spanx*IW).toFixed(1);
  const Y = v  => (MT + (1-v)*IH).toFixed(1);
  const tip = p => `<title>${p.b}bit &#215; ${p.d}d &#183; ${fmtB(p.bytes)} &#183; ${p.v.toFixed(3)}</title>`;
  let g = `<svg viewBox="0 0 ${CW} ${CH}" class="chart">`;
  for (let t=0;t<=10;t+=2){ const yy=Y(t/10);                                                       // y grid + labels
    g += `<line class="grid" x1="${ML}" y1="${yy}" x2="${CW-MR}" y2="${yy}"/><text class="ax" x="${ML-6}" y="${+yy+3}" text-anchor="end">${(t/10).toFixed(1)}</text>`; }
  for (let e=Math.ceil(lo); e<=Math.floor(hi); e++){ const by=2**e, xx=X(by);                        // x grid + labels (powers of two bytes)
    g += `<line class="grid" x1="${xx}" y1="${MT}" x2="${xx}" y2="${MT+IH}"/><text class="ax" x="${xx}" y="${MT+IH+16}" text-anchor="middle">${fmtB(by)}</text>`; }
  P.forEach(p => g += `<circle cx="${X(p.bytes)}" cy="${Y(p.v)}" r="2.4" fill="${color(p.v)}" opacity="0.45">${tip(p)}</circle>`);
  const fr = P.filter(p=>p.pa).sort((a,b)=>a.bytes-b.bytes);                                          // pareto frontier line + dots
  g += `<polyline class="front" points="${fr.map(p=>X(p.bytes)+','+Y(p.v)).join(' ')}"/>`;
  fr.forEach(p => g += `<circle cx="${X(p.bytes)}" cy="${Y(p.v)}" r="4" fill="${color(p.v)}" stroke="#fff" stroke-width="1.5">${tip(p)}</circle>`);
  g += `<text class="axt" x="${ML+IW/2}" y="${CH-4}" text-anchor="middle">bytes / vector (log)</text>`;
  g += `<text class="axt" transform="translate(14,${MT+IH/2}) rotate(-90)" text-anchor="middle">${METRICS[st.me]}</text></svg>`;
  document.getElementById('chart').innerHTML = g;
}
buildSwitch(document.getElementById('sw-ds'), uniq('ds'), st.ds, k => { st.ds=k; render(); });
buildSwitch(document.getElementById('sw-ty'), uniq('ty'), st.ty, k => { st.ty=k; render(); });
buildSwitch(document.getElementById('sw-ro'), uniq('ro'), st.ro, k => { st.ro=k; render(); });
buildSwitch(document.getElementById('sw-me'), Object.keys(METRICS), st.me, k => { st.me=k; render(); }, METRICS);
render();
// hover a cell -> fade every cell whose value is below it, so all cells with >= recall stay highlighted
const _tbl = document.getElementById('tbl');
const _legval = document.getElementById('legval');
// dim cells below the threshold; among those at/above it, outline the single best recall-per-byte cell
const applyThreshold = v => {
  let best = null, bestRatio = -Infinity;
  _tbl.querySelectorAll('td.cell').forEach(td => {
    const val = +td.dataset.v;
    td.classList.toggle('dim', val < v);
    td.classList.remove('best');
    if (val >= v) { const ratio = val / +td.dataset.bytes; if (ratio > bestRatio) { bestRatio = ratio; best = td; } }
  });
  if (best) best.classList.add('best');
};
const _clear = () => { _tbl.querySelectorAll('td.cell.dim,td.cell.best').forEach(td => td.classList.remove('dim','best')); _legval.textContent = ''; };
// hover a cell -> highlight cells with >= its recall
_tbl.addEventListener('mouseover', e => { const c = e.target.closest('td.cell'); if (!c) { _clear(); return; } applyThreshold(+c.dataset.v); });
_tbl.addEventListener('mouseleave', _clear);
// hover the legend bar -> highlight cells with >= the recall at the pointer position
const _bar = document.getElementById('bar');
_bar.addEventListener('mousemove', e => {
  const r = _bar.getBoundingClientRect();
  const v = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  applyThreshold(v); _legval.textContent = '≥ ' + v.toFixed(2);
});
_bar.addEventListener('mouseleave', _clear);
</script></body></html>
'''
open(os.path.join(BASE, 'site/recall.html'), 'w').write(HTML.replace('__DATA__', data))
print('wrote results/recall.json (%d rows) and site/recall.html' % len(rows))
