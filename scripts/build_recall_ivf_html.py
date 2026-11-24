#!/usr/bin/env python3
# Build results/recall_ivf.json + recall_ivf.html (self-contained heatmap viewer)
# from results/recall_ivf.csv (written by scripts/recall_ivf_bench.py).
#
# Sibling of build_recall_codecs_html.py, same palette and controls. Its own page because IVF is measured on
# the full table, not the ~100k pools the codec page uses, so the numbers must not sit side by side.
# Grid: nprobe (rows) x multiplier (columns), with k on a switch.
import csv, json, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for r in csv.DictReader(open(os.path.join(BASE, 'results/recall_ivf.csv'))):
    rows.append({'np': int(r['nprobe']), 'nc': int(r['clusters']), 'k': int(r['k']), 'm': int(r['multiplier']),
                 'ca': int(r['candidates']), 'rr': int(r['rows_read']), 'tr': int(r['table_rows']),
                 'rf': float(r['read_frac']), 'qz': int(r['quantized']),
                 'recall_at_k': float(r['recall_at_k']), 'recall10': float(r['recall10'])})
data = json.dumps(rows, separators=(',', ':'))
json.dump(rows, open(os.path.join(BASE, 'results/recall_ivf.json'), 'w'), separators=(',', ':'))
nc, tr = rows[0]['nc'], rows[0]['tr']

HTML = r'''<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>IVF recall</title>
<style>
  body { background:#111; color:#ddd; font:16px system-ui,-apple-system,sans-serif; margin:0; padding:1em 1.2em; }
  a { color:#ff0; }
  h1 { font-size:19px; color:#ff0; margin:.2em 0 .1em; }
  .sub { color:#888; font-size:14px; margin-bottom:.6em; max-width:74em; line-height:1.45; }
  .ctl { display:flex; flex-wrap:wrap; gap:.35em 1.4em; align-items:center; margin:.5em 0 1em; }
  .ctl > div { display:flex; align-items:center; gap:.3em; }
  .ctl label { color:#999; font-size:13px; text-transform:uppercase; letter-spacing:.04em; }
  .opt { padding:.16em .55em; background:#2b2b2b; border:1px solid #000; cursor:pointer; user-select:none; font-size:14px; border-radius:3px; }
  .opt:hover { background:#ff0; color:#000; }
  .opt.sel { background:#fff; color:#000; }
  table { border-collapse:collapse; }
  caption { caption-side:top; text-align:left; color:#8bc34a; font-family:monospace; margin-bottom:.45em; }
  th, td { padding:.28em .5em; text-align:center; font-variant-numeric:tabular-nums; font-size:14px; }
  th { color:#9a9a9a; font-weight:normal; }
  th.cor { color:#ff0; text-align:right; white-space:nowrap; }
  th.grp { color:#29b6f6; border-bottom:1px solid #2a2a2a; }
  td.cell { color:#000; font-weight:600; min-width:60px; border:1px solid #111; transition:opacity .08s; line-height:1.15; }
  td.cell.dim { opacity:.15; }         /* on hover: cells below the pointed value fade, so >= cells stand out */
  td.cell .sz { display:block; font-size:11px; font-weight:400; opacity:.6; }   /* share of the table read */
  td.na { color:#555; }
  .legend { display:flex; align-items:center; gap:.4em; margin-top:1em; color:#999; font-size:13px; }
  .bar { width:220px; height:14px; border-radius:2px; cursor:crosshair;
         background:linear-gradient(90deg,hsl(0,75%,50%),hsl(60,75%,50%),hsl(120,75%,50%)); }
  .legval { color:#ff0; font-family:monospace; min-width:4em; }
  .foot { margin-top:2.2em; color:#888; font-size:13px; }
  .warn { color:#ff7043; font-size:13px; margin-top:.8em; }
</style></head><body>
<h1>IVF recall &mdash; Photos / siglip2</h1>
<div class=sub><a href="index.html">&#8592; Embeddings Explorer</a> &middot; <a href="recall_codecs.html">Codec recall &#8599;</a> &middot; <a href="recall.html">QBit recall &#8599;</a><br>
IVF over the full __TR__M-row table: find the <i>nprobe</i> centroids (of __NC__) nearest the query, then scan only
their clusters' RaBitQ codes, keep k&nbsp;&times;&nbsp;multiplier candidates and rescore them against the
full-precision vectors. Centroids are found through their HNSW index, as in the Explorer. That index leaves ~2.5%
of them unreachable, which costs ~4 points of recall@100; an exact centroid scan (<code>use_skip_indexes&nbsp;=&nbsp;0</code>)
reaches 0.955 at k&nbsp;=&nbsp;100, nprobe&nbsp;=&nbsp;64, &times;10 but takes ~160&nbsp;ms longer per query.<br>
Measured against exact-L2 ground truth over every row, 100 query photos. Green = 1.0, red = 0.0. Second line in each
cell is the share of the table read. Not comparable with the codec page, which uses ~100k-row pools: finding the true
top-k among 99M rows is harder. The Explorer uses nprobe&nbsp;=&nbsp;64, k&nbsp;=&nbsp;100, multiplier&nbsp;&times;10.</div>
<div class=ctl>
  <div><label>nearest (k)</label><span id=sw-k></span></div>
  <div><label>metric</label><span id=sw-me></span></div>
</div>
<div id=tbl></div>
<div class=legend>0.0 <span class=bar id=bar></span> 1.0 <span id=legval class=legval></span></div>
<div id=warn class=warn></div>
<div class=foot><a href="https://github.com/ClickHouse/embeddings" target=_blank rel=noopener>About</a> | &copy; Alexey Milovidov, ClickHouse, Inc. (data: Multimedia Commons)</div>
<script>
const DATA = __DATA__;
const METRICS = { recall_at_k:'Recall@k', recall10:'Recall@10' };
const NPS   = [...new Set(DATA.map(r => r.np))].sort((a,b)=>a-b);
const MULTS = [...new Set(DATA.map(r => r.m))].sort((a,b)=>a-b);
const KS    = [...new Set(DATA.map(r => r.k))].sort((a,b)=>a-b);
let st = { k:(KS.includes(100) ? 100 : KS[0]), me:'recall_at_k' };
const color = v => `hsl(${(v*120).toFixed(0)},75%,50%)`;
const pct = f => (100*f < 1 ? (100*f).toFixed(2) : (100*f).toFixed(1)) + '%';
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
  const m = {}; DATA.filter(r => r.k === st.k).forEach(r => m[`${r.np}|${r.m}`] = r);
  let h = `<table><caption>${METRICS[st.me]} &middot; k = ${st.k} &middot; 100 queries</caption>`;
  h += `<tr><th></th><th class=grp colspan="${MULTS.length}">multiplier</th></tr>`;
  h += '<tr><th class=cor>nprobe</th>' + MULTS.map(x => `<th>&times;${x}</th>`).join('') + '</tr>';
  for (const np of NPS) {
    h += `<tr><th class=cor>${np}</th>`;
    for (const x of MULTS) {
      const r = m[`${np}|${x}`];
      if (!r) { h += '<td class=na>&mdash;</td>'; continue; }
      const v = r[st.me];
      h += `<td class=cell data-v="${v}" style="background:${color(v)}"`
         + ` title="nprobe ${np} of ${r.nc} · k=${r.k} · multiplier ${x} · ${r.ca} candidates · read ${r.rr.toLocaleString('en-US')} rows (${pct(r.rf)})${r.qz?'':' · NOT quantized'}">`
         + `${v.toFixed(3)}${r.qz ? '' : ' &#9888;'}<span class=sz>${pct(r.rf)}</span></td>`;
    }
    h += '</tr>';
  }
  document.getElementById('tbl').innerHTML = h + '</table>';
  const bad = DATA.filter(r => !r.qz && r.k === st.k);
  document.getElementById('warn').innerHTML = bad.length
    ? `&#9888; ${bad.length} config(s) did not use the quantized codes (full-precision fallback) &mdash; their recall is not meaningful.` : '';
  alignLeft();
}
function alignLeft(){
  const t = document.querySelector('#tbl table'); if (!t || !t.rows[1] || !t.rows[1].cells[1]) return;
  document.querySelector('.legend').style.paddingLeft = (t.rows[1].cells[1].getBoundingClientRect().left - t.getBoundingClientRect().left) + 'px';
}
window.addEventListener('resize', alignLeft);
buildSwitch(document.getElementById('sw-k'), KS, st.k, v => { st.k=+v; render(); });
buildSwitch(document.getElementById('sw-me'), Object.keys(METRICS), st.me, k => { st.me=k; render(); }, METRICS);
render();
const _tbl = document.getElementById('tbl'), _legval = document.getElementById('legval'), _bar = document.getElementById('bar');
const applyThreshold = v => _tbl.querySelectorAll('td.cell').forEach(td => td.classList.toggle('dim', +td.dataset.v < v));
const _clear = () => { _tbl.querySelectorAll('td.cell.dim').forEach(td => td.classList.remove('dim')); _legval.textContent = ''; };
_tbl.addEventListener('mouseover', e => { const c = e.target.closest('td.cell'); if (!c) { _clear(); return; } applyThreshold(+c.dataset.v); });
_tbl.addEventListener('mouseleave', _clear);
_bar.addEventListener('mousemove', e => {
  const r = _bar.getBoundingClientRect(), v = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  applyThreshold(v); _legval.textContent = '≥ ' + v.toFixed(2);
});
_bar.addEventListener('mouseleave', _clear);
</script></body></html>
'''
html = (HTML.replace('__DATA__', data).replace('__TR__', f'{tr / 1e6:.1f}')
            .replace('__NC__', f'{nc:,}'))
open(os.path.join(BASE, 'recall_ivf.html'), 'w').write(html)
print('wrote results/recall_ivf.json (%d rows) and recall_ivf.html' % len(rows))
