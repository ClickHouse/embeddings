#!/usr/bin/env python3
# Merge one or more recall_results_<label>.csv (from recall_scale_bench.py) into results/recall.csv,
# which carries a leading `sample` column. Existing rows lacking one are treated as sample=100k.
# Dedupe by full config key (sample,dataset,type,rotation,bits,dims); last file wins. Rows sorted
# for stable diffs. Usage:  python3 scripts/recall_merge.py /tmp/recall_results_1m.csv [...]
import csv, os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(BASE, 'results/recall.csv')
COLS = ['sample','dataset','type','rotation','bits','dims','recall10','recall100','recall10in100']
ORDER = {'100k':0, '1m':1, '10m':2, 'full':3}
def key(r): return (r['sample'], r['dataset'], r['type'], r['rotation'], r['bits'], r['dims'])

rows = {}
if os.path.exists(DEST):
    for r in csv.DictReader(open(DEST)):
        if not r.get('sample'): r['sample'] = '100k'
        rows[key(r)] = {c: r.get(c, '') for c in COLS}
for path in sys.argv[1:]:
    for r in csv.DictReader(open(path)):
        if not r.get('recall100'): continue
        rows[key(r)] = {c: r.get(c, '') for c in COLS}

out = sorted(rows.values(), key=lambda r: (ORDER.get(r['sample'], 9), r['dataset'], r['type'], r['rotation'], int(r['bits']), int(r['dims'])))
with open(DEST, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
    for r in out: w.writerow(r)
from collections import Counter
cnt = Counter(r['sample'] for r in out)
print(f"wrote {DEST}: {len(out)} rows; per-sample counts = " +
      ", ".join(f"{s}={cnt[s]}" for s in sorted(cnt, key=lambda s: ORDER.get(s, 9))))
