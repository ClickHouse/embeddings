#!/usr/bin/env python3
# Build results/recall_codecs.json + recall_codecs.html (self-contained heatmap viewer)
# from results/recall_codecs.csv (written by scripts/recall_codecs_bench.py).
#
# Sibling of build_recall_html.py. Same palette and controls on purpose, but a different grid: the QBit
# page is bits x dims for one (model, type, rotation); this one is model x (codec, multiplier) with k on a
# switch, because the codec reps have no bit-plane or dim truncation -- their knobs are k and the
# candidate over-fetch, exactly the two chips the Explorer shows.
import csv, json, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for r in csv.DictReader(open(os.path.join(BASE, 'results/recall_codecs.csv'))):
    if not r['recall_at_k']:
        continue
    rows.append({'co': r['corpus'], 'ds': r['dataset'], 'cd': r['codec'],
                 'k': int(r['k']), 'm': int(r['multiplier']), 'ca': int(r['candidates']),
                 'dim': int(r['dim']), 'cb': int(r['code_bytes']), 'qz': int(r['quantized']),
                 'recall_at_k': float(r['recall_at_k']), 'recall10': float(r['recall10'])})
data = json.dumps(rows, separators=(',', ':'))
json.dump(rows, open(os.path.join(BASE, 'results/recall_codecs.json'), 'w'), separators=(',', ':'))

HTML = r'''<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Quantized codec recall</title>
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
  th.cs  { color:#666; font-size:12px; text-align:left; padding-top:.9em; text-transform:uppercase; letter-spacing:.05em; }
  td.cell { color:#000; font-weight:600; min-width:52px; border:1px solid #111; transition:opacity .08s; line-height:1.15; }
  td.cell.dim { opacity:.15; }         /* on hover: cells below the pointed value fade, so >= cells stand out */
  td.cell .sz { display:block; font-size:11px; font-weight:400; opacity:.6; }   /* code bytes/vector, second line */
  td.cell.best { outline:2px solid #fff; outline-offset:-2px; box-shadow:0 0 12px 3px rgba(255,255,255,.9);
                 filter:brightness(1.18); position:relative; z-index:2; }
  td.na { color:#555; }
  .legend { display:flex; align-items:center; gap:.4em; margin-top:1em; color:#999; font-size:13px; }
  .bar { width:220px; height:14px; border-radius:2px; cursor:crosshair;
         background:linear-gradient(90deg,hsl(0,75%,50%),hsl(60,75%,50%),hsl(120,75%,50%)); }
  .legval { color:#ff0; font-family:monospace; min-width:4em; }
  .foot { margin-top:2.2em; color:#888; font-size:13px; }
  .foot a { color:#ff0; }
  .warn { color:#ff7043; font-size:13px; margin-top:.8em; }
</style></head><body>
<h1>Quantized codec recall &mdash; RaBitQ vs TurboQuant</h1>
<div class=sub><a href="index.html">&#8592; Embeddings Explorer</a> &middot; <a href="recall.html">QBit recall &#8599;</a><br>
Brute-force two-stage search over an <code>Array(BFloat16) CODEC(Quantized(&hellip;))</code> column: scan the
quantized codes, keep k&nbsp;&times;&nbsp;multiplier candidates, then rescore those against the full-precision
vectors. Measured against exact-cosine ground truth on a ~100k-row pool (deduplicated by vector), 100 random
queries. Green = 1.0, red = 0.0. Second line in each cell is the code size per vector (1&nbsp;bit/dim for
RaBitQ, 2&nbsp;bits/dim for TurboQuant).<br>
<b>Recall follows the candidate pool, k&nbsp;&times;&nbsp;multiplier &mdash; not the multiplier alone.</b>
k&nbsp;=&nbsp;10&nbsp;&times;10 and k&nbsp;=&nbsp;100&nbsp;&times;1 both fetch 100 candidates and score alike;
k&nbsp;=&nbsp;10&nbsp;&times;1 fetches 10 and scores far worse. So read the <i>k</i> rows separately &mdash;
the Explorer's default is k&nbsp;=&nbsp;100, multiplier&nbsp;&times;1.</div>
<div class=ctl>
  <div><label>nearest (k)</label><span id=sw-k></span></div>
  <div><label>metric</label><span id=sw-me></span></div>
</div>
<div id=tbl></div>
<div class=legend>0.0 <span class=bar id=bar></span> 1.0 <span id=legval class=legval></span></div>
<div id=warn class=warn></div>
<div class=foot><a href="https://github.com/ClickHouse/embeddings" target=_blank rel=noopener>About</a> | &copy; Alexey Milovidov, ClickHouse, Inc. (data: Multimedia Commons / WonderfulWeb / Hacker News)</div>
<script>
const DATA = __DATA__;
const METRICS = { recall_at_k:'Recall@k', recall10:'Recall@10' };
const CORPORA = { photos:'Photos', web:'Web', hn:'HackerNews' };
const CODECS = ['rabitq', 'turboquant'];
const CODEC_LABEL = { rabitq:'RaBitQ', turboquant:'TurboQuant' };
const MULTS = [...new Set(DATA.map(r => r.m))].sort((a,b)=>a-b);
const KS    = [...new Set(DATA.map(r => r.k))].sort((a,b)=>a-b);
let st = { k:(KS.includes(100) ? 100 : KS[0]), me:'recall_at_k' };
const color = v => v==null ? '#222' : `hsl(${(v*120).toFixed(0)},75%,50%)`;
const fmtB = n => n < 1024 ? n+'B' : (n/1024).toFixed(1).replace(/\.0$/,'')+'K';
const cellKey = (co,ds,cd,k,m) => `${co}|${ds}|${cd}|${k}|${m}`;
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
  const m = {}; DATA.forEach(r => m[cellKey(r.co,r.ds,r.cd,r.k,r.m)] = r);
  // model rows, grouped by corpus, in the order the corpora appear in CORPORA
  const models = [];
  for (const co of Object.keys(CORPORA))
    for (const ds of [...new Set(DATA.filter(r => r.co===co).map(r => r.ds))]) models.push([co, ds]);

  let h = `<table><caption>${METRICS[st.me]} &middot; k = ${st.k} &middot; ~100k-row pool &middot; 100 queries</caption>`;
  h += '<tr><th></th>' + CODECS.map(c => `<th class=grp colspan="${MULTS.length}">${CODEC_LABEL[c]}</th>`).join('') + '</tr>';
  h += '<tr><th class=cor>model&nbsp;\\&nbsp;multiplier</th>'
     + CODECS.map(() => MULTS.map(x => `<th>&times;${x}</th>`).join('')).join('') + '</tr>';
  let lastCo = null;
  for (const [co, ds] of models) {
    if (co !== lastCo) { h += `<tr><th class=cs colspan="${1 + CODECS.length*MULTS.length}">${CORPORA[co]}</th></tr>`; lastCo = co; }
    h += `<tr><th class=cor>${ds}</th>`;
    for (const cd of CODECS) for (const x of MULTS) {
      const r = m[cellKey(co,ds,cd,st.k,x)];
      if (!r) { h += '<td class=na>&mdash;</td>'; continue; }
      const v = r[st.me];
      const flag = r.qz ? '' : ' &#9888;';
      h += `<td class=cell data-k="${cellKey(co,ds,cd,st.k,x)}" data-v="${v}" data-cb="${r.cb}"`
         + ` style="background:${color(v)}"`
         + ` title="${CORPORA[co]} / ${ds} · ${CODEC_LABEL[cd]} · k=${r.k} · multiplier ${x} · ${r.ca} candidates · ${r.dim}d · ${fmtB(r.cb)}/vector${r.qz?'':' · NOT quantized'}">`
         + `${v.toFixed(3)}${flag}<span class=sz>${fmtB(r.cb)}</span></td>`;
    }
    h += '</tr>';
  }
  h += '</table>';
  document.getElementById('tbl').innerHTML = h;
  const bad = DATA.filter(r => !r.qz && r.k === st.k);
  document.getElementById('warn').innerHTML = bad.length
    ? `&#9888; ${bad.length} config(s) did not use the quantized codes (full-precision fallback) &mdash; their recall is not meaningful.`
    : '';
  alignLeft();
}
buildSwitch(document.getElementById('sw-k'), KS, st.k, v => { st.k=+v; render(); });
buildSwitch(document.getElementById('sw-me'), Object.keys(METRICS), st.me, k => { st.me=k; render(); }, METRICS);
render();
const _tbl = document.getElementById('tbl');
const _legval = document.getElementById('legval');
// align the legend left edge with the first VALUE column (past the row-header column)
function alignLeft(){
  const t = document.querySelector('#tbl table'); if (!t || !t.rows[1] || !t.rows[1].cells[1]) return;
  const pad = (t.rows[1].cells[1].getBoundingClientRect().left - t.getBoundingClientRect().left) + 'px';
  document.querySelector('.legend').style.paddingLeft = pad;
}
window.addEventListener('resize', alignLeft);
// dim cells below the threshold; among those at/above it, outline the single best recall-per-code-byte cell
const applyThreshold = v => {
  let best = null, bestRatio = -Infinity;
  _tbl.querySelectorAll('td.cell').forEach(td => {
    const val = +td.dataset.v;
    td.classList.toggle('dim', val < v);
    td.classList.remove('best');
    if (val >= v) { const ratio = val / +td.dataset.cb; if (ratio > bestRatio) { bestRatio = ratio; best = td; } }
  });
  if (best) best.classList.add('best');
};
const _clear = () => { _tbl.querySelectorAll('td.cell.dim,td.cell.best').forEach(td => td.classList.remove('dim','best')); _legval.textContent = ''; };
_tbl.addEventListener('mouseover', e => { const c = e.target.closest('td.cell'); if (!c) { _clear(); return; } applyThreshold(+c.dataset.v); });
_tbl.addEventListener('mouseleave', _clear);
const _bar = document.getElementById('bar');
_bar.addEventListener('mousemove', e => {
  const r = _bar.getBoundingClientRect();
  const v = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  applyThreshold(v); _legval.textContent = '≥ ' + v.toFixed(2);
});
_bar.addEventListener('mouseleave', _clear);
</script></body></html>
'''
open(os.path.join(BASE, 'recall_codecs.html'), 'w').write(HTML.replace('__DATA__', data))
print('wrote results/recall_codecs.json (%d rows) and recall_codecs.html' % len(rows))
