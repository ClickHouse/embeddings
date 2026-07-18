#!/usr/bin/env python3
# Run the full recall grid against a given sample size and tag each row with it.
#   python3 scripts/recall_scale_bench.py <label> [workers]     e.g.  1m 8   |   100k 8
# label '100k' uses the original unsuffixed bench.samp_<ds>/gt_<ds>; any other label uses
# bench.samp_<ds>_<label>/gt_<ds>_<label> built by recall_scale_setup.py. Queries (bench.q_<ds>)
# are shared. Writes /tmp/recall_results_<label>.csv with a leading `sample` column.
import os, csv, time, http.client, ssl, threading, sys
from concurrent.futures import ThreadPoolExecutor

HOST='hvdvsqo23t.us-east-2.aws.clickhouse-staging.com'; PORT=8443
PW=os.environ['CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD']
LABEL   = sys.argv[1] if len(sys.argv) > 1 else '1m'
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SUF = '' if LABEL == '100k' else '_' + LABEL
FULL = LABEL == 'full'
SRC = (lambda ds: f"mmcommons.{ds}") if FULL else (lambda ds: f"bench.samp_{ds}{SUF}")
GT  = (lambda ds: f"bench.gt_{ds}_full") if FULL else (lambda ds: f"bench.gt_{ds}{SUF}")
# NB: NO parallel replicas — for this CROSS JOIN + transposed-distance + groupArraySorted pattern the PR
# coordinator overhead dominates (heavy config: 46.9s with PR=3 vs 7.7s single-node). Rely on worker
# concurrency instead: each query runs single-node and the HTTP LB spreads workers across the 3 replicas.
SETTINGS = '&allow_experimental_qbit_type=1&max_execution_time=1800&max_memory_usage=40000000000'

DATASETS={'emb_siglip2':{'original':1152,'rotated':2048},'emb_clip':{'original':768,'rotated':768},'emb_nomic':{'original':768,'rotated':768}}
TYPES={'BFloat16':('cosineDistanceTransposed',16),'Int8':('cosineDistanceTransposedQuantized',8)}
COLREF={('BFloat16','original'):('embedding_strided','ref_orig'),('BFloat16','rotated'):('embedding_rotated','ref_rot'),
        ('Int8','original'):('embedding_int','ref_orig'),('Int8','rotated'):('embedding_rotated_int','ref_rot')}

configs=[]
for ds,dd in DATASETS.items():
  for typ,(fn,maxbits) in TYPES.items():
    for rot in ('original','rotated'):
      col,ref=COLREF[(typ,rot)]; maxdim=dd[rot]
      for bits in range(1,maxbits+1):
        for dims in range(128,maxdim+1,128):
          configs.append((ds,typ,rot,col,ref,fn,bits,dims))
print(f"label={LABEL} suffix='{SUF}' configs={len(configs)} workers={WORKERS}",flush=True)

def sql(ds,col,ref,fn,bits,dims):
  # groupArraySorted(100): bounded per-query top-100 heap, single table scan (no giant sort) -> scales to full
  return (f"SELECT round(avg(r10),4),round(avg(r100),4),round(avg(r10in100),4) FROM ("
    f"SELECT length(arrayIntersect(g.gt10,arraySlice(a.ap,1,10)))/10. r10,"
    f"length(arrayIntersect(g.gt100,a.ap))/100. r100,length(arrayIntersect(g.gt10,a.ap))/10. r10in100 "
    f"FROM (SELECT q_idx,arrayMap(t->t.2,groupArraySorted(100)(tuple(dist,md5))) ap FROM ("
    f"SELECT q.q_idx q_idx,s.md5 md5,{fn}(s.{col},arraySlice(q.{ref},1,{dims}),{bits},{dims}) dist "
    f"FROM {SRC(ds)} s CROSS JOIN bench.q_{ds} q WHERE s.md5!=q.q_md5"
    f") GROUP BY q_idx) a JOIN {GT(ds)} g ON a.q_idx=g.q_idx) FORMAT TSV")

tls=ssl.create_default_context(); local=threading.local()
def run(cfg):
  ds,typ,rot,col,ref,fn,bits,dims=cfg
  body=sql(ds,col,ref,fn,bits,dims).encode()
  for att in range(4):
    try:
      if not hasattr(local,'c'): local.c=http.client.HTTPSConnection(HOST,PORT,context=tls,timeout=1850)
      local.c.request('POST','/?'+SETTINGS.lstrip('&'),body=body,headers={'X-ClickHouse-User':'default','X-ClickHouse-Key':PW})
      r=local.c.getresponse(); data=r.read().decode()
      if r.status!=200: raise Exception(data[:150])
      a,b,c=data.strip().split('\t')
      return (LABEL,ds,typ,rot,bits,dims,float(a),float(b),float(c),'')
    except Exception as e:
      try: local.c.close(); del local.c
      except: pass
      if att==3: return (LABEL,ds,typ,rot,bits,dims,None,None,None,str(e)[:150])
      time.sleep(2)

out=f"/tmp/recall_results_{LABEL}.csv"
HDR=['sample','dataset','type','rotation','bits','dims','recall10','recall100','recall10in100']
# resume: skip configs already written with a real recall (crash-safe over long runs; re-run to continue)
done_keys=set()
if os.path.exists(out):
  for r in csv.DictReader(open(out)):
    if r.get('recall100') not in (None,''): done_keys.add((r['dataset'],r['type'],r['rotation'],r['bits'],r['dims']))
todo=[c for c in configs if (c[0],c[1],c[2],str(c[6]),str(c[7])) not in done_keys]
print(f"resume: {len(done_keys)} done, {len(todo)} to run",flush=True)
f=open(out,'a',newline=''); w=csv.writer(f)
if not done_keys: w.writerow(HDR); f.flush()
done=0; fails=0; t0=time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
  for res in ex.map(run,todo):
    w.writerow(res[:9]); f.flush(); done+=1                     # persist each result immediately
    if res[6] is None: fails+=1; print("FAIL",res[:6],res[9],flush=True)
    if done%50==0: print(f"{done}/{len(todo)} {time.time()-t0:.0f}s",flush=True)
f.close()
print(f"DONE {done} configs in {time.time()-t0:.0f}s, failures={fails} -> {out}",flush=True)
