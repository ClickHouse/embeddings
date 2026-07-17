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
  body { background:#111; color:#ddd; font:13px system-ui,-apple-system,sans-serif; margin:0; padding:1em 1.2em; }
  h1 { font-size:16px; color:#ffd54f; margin:.2em 0 .1em; }
  .sub { color:#888; font-size:12px; margin-bottom:.6em; }
  .ctl { display:flex; flex-wrap:wrap; gap:.35em 1.4em; align-items:center; margin:.5em 0 1em; }
  .ctl > div { display:flex; align-items:center; gap:.3em; }
  .ctl label { color:#999; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .opt { padding:.16em .55em; background:#2b2b2b; border:1px solid #000; cursor:pointer; user-select:none; font-size:12px; border-radius:3px; }
  .opt:hover { background:#ffd54f; color:#000; }
  .opt.sel { background:#fff; color:#000; }
  table { border-collapse:collapse; }
  caption { caption-side:top; text-align:left; color:#8bc34a; font-family:monospace; margin-bottom:.45em; }
  th, td { padding:.28em .5em; text-align:center; font-variant-numeric:tabular-nums; font-size:12px; }
  th { color:#9a9a9a; font-weight:normal; }
  th.cor { color:#ffd54f; }
  td.cell { color:#000; font-weight:600; min-width:44px; border:1px solid #111; transition:opacity .08s; line-height:1.15; }
  td.cell.dim { opacity:.15; }         /* on hover: cells below the pointed value fade, so >= cells stand out */
  td.cell .sz { display:block; font-size:9px; font-weight:400; opacity:.6; }   /* bytes/vector, second line */
  .legend { display:flex; align-items:center; gap:.4em; margin-top:1em; color:#999; font-size:11px; }
  .bar { width:180px; height:12px; border-radius:2px;
         background:linear-gradient(90deg,hsl(0,75%,50%),hsl(60,75%,50%),hsl(120,75%,50%)); }
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
<div class=legend>0.0 <span class=bar></span> 1.0</div>
<script>
const DATA = __DATA__;
const METRICS = { recall100:'Recall@100', recall10:'Recall@10', recall10in100:'Recall 10-in-100' };
let st = { ds:'siglip2', ty:'Int8', ro:'rotated', me:'recall100' };
const uniq = k => [...new Set(DATA.map(r => r[k]))];
const color = v => v==null ? '#222' : `hsl(${(v*120).toFixed(0)},75%,50%)`;
const bytesPerVec = (b,d) => { const n = b*d/8; return n < 1024 ? n+'B' : (n/1024).toFixed(1).replace(/\.0$/,'')+'K'; };
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
  let h = `<table><caption>${st.ds} &middot; ${st.ty} &middot; ${st.ro} &middot; ${METRICS[st.me]}</caption>`;
  h += '<tr><th class=cor>bits&nbsp;\\&nbsp;dims</th>' + dims.map(d => `<th>${d}</th>`).join('') + '</tr>';
  for (const b of bits) {
    h += `<tr><th class=cor>${b}</th>` + dims.map(d => {
      const v = m[b + '_' + d];
      return v==null ? '<td></td>' : `<td class=cell data-v="${v}" style="background:${color(v)}" title="bits=${b} dims=${d}">${v.toFixed(2)}<span class=sz>${bytesPerVec(b,d)}</span></td>`;
    }).join('') + '</tr>';
  }
  h += '</table>';
  document.getElementById('tbl').innerHTML = h;
}
buildSwitch(document.getElementById('sw-ds'), uniq('ds'), st.ds, k => { st.ds=k; render(); });
buildSwitch(document.getElementById('sw-ty'), uniq('ty'), st.ty, k => { st.ty=k; render(); });
buildSwitch(document.getElementById('sw-ro'), uniq('ro'), st.ro, k => { st.ro=k; render(); });
buildSwitch(document.getElementById('sw-me'), Object.keys(METRICS), st.me, k => { st.me=k; render(); }, METRICS);
render();
// hover a cell -> fade every cell whose value is below it, so all cells with >= recall stay highlighted
const _tbl = document.getElementById('tbl');
const _clear = () => _tbl.querySelectorAll('td.cell.dim').forEach(td => td.classList.remove('dim'));
_tbl.addEventListener('mouseover', e => {
  const c = e.target.closest('td.cell');
  if (!c) { _clear(); return; }
  const v = +c.dataset.v;
  _tbl.querySelectorAll('td.cell').forEach(td => td.classList.toggle('dim', (+td.dataset.v) < v));
});
_tbl.addEventListener('mouseleave', _clear);
</script></body></html>
'''
open(os.path.join(BASE, 'site/recall.html'), 'w').write(HTML.replace('__DATA__', data))
print('wrote results/recall.json (%d rows) and site/recall.html' % len(rows))
