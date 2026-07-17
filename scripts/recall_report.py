import csv
rows=[r for r in csv.DictReader(open('/tmp/recall_results.csv')) if r['recall100']]
DS=['emb_siglip2','emb_clip','emb_nomic']; TY=['BFloat16','Int8']; RO=['original','rotated']
def num(x): return float(x)
def pivot(ds,typ,rot,metric):
    sub=[r for r in rows if r['dataset']==ds and r['type']==typ and r['rotation']==rot]
    if not sub: return None
    bits=sorted({int(r['bits']) for r in sub}); dims=sorted({int(r['dims']) for r in sub})
    m={(int(r['bits']),int(r['dims'])):num(r[metric]) for r in sub}
    out=['| bits\\dims | '+' | '.join(str(d) for d in dims)+' |','|'+'---|'*(len(dims)+1)]
    for b in bits:
        out.append(f'| **{b}** | '+' | '.join(f'{m[(b,d)]:.2f}' if (b,d) in m else '·' for d in dims)+' |')
    return '\n'.join(out)
metric='recall100'
import sys
metric=sys.argv[1] if len(sys.argv)>1 else 'recall100'
md=[f'# Recall benchmark ({metric}) — 100k-row sample, 20 queries\n']
for ds in DS:
    md.append(f'\n## {ds}')
    for typ in TY:
        for rot in RO:
            p=pivot(ds,typ,rot,metric)
            if p: md.append(f'\n**{typ} · {rot}**  (rows=bits, cols=dims)\n\n'+p)
print('\n'.join(md))
