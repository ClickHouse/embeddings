import os, csv, time, http.client, ssl, threading
from concurrent.futures import ThreadPoolExecutor

HOST='hvdvsqo23t.us-east-2.aws.clickhouse-staging.com'; PORT=8443
PW=os.environ['CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD']
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
print("configs:",len(configs),flush=True)

def sql(ds,col,ref,fn,bits,dims):
  return (f"SELECT round(avg(r10),4),round(avg(r100),4),round(avg(r10in100),4) FROM ("
    f"SELECT length(arrayIntersect(g.gt10,arraySlice(a.ap,1,10)))/10. r10,"
    f"length(arrayIntersect(g.gt100,a.ap))/100. r100,length(arrayIntersect(g.gt10,a.ap))/10. r10in100 "
    f"FROM (SELECT q_idx,arrayMap(t->t.2,arraySort(t->t.1,groupArray((dist,md5)))) ap FROM ("
    f"SELECT q.q_idx q_idx,s.md5 md5,{fn}(s.{col},arraySlice(q.{ref},1,{dims}),{bits},{dims}) dist "
    f"FROM bench.samp_{ds} s CROSS JOIN bench.q_{ds} q WHERE s.md5!=q.q_md5 "
    f"ORDER BY q_idx,dist ASC LIMIT 100 BY q_idx) GROUP BY q_idx) a JOIN bench.gt_{ds} g ON a.q_idx=g.q_idx) FORMAT TSV")

tls=ssl.create_default_context(); local=threading.local()
def run(cfg):
  ds,typ,rot,col,ref,fn,bits,dims=cfg
  body=sql(ds,col,ref,fn,bits,dims).encode()
  for att in range(4):
    try:
      if not hasattr(local,'c'): local.c=http.client.HTTPSConnection(HOST,PORT,context=tls,timeout=200)
      local.c.request('POST','/?allow_experimental_qbit_type=1&max_execution_time=180',body=body,
                      headers={'X-ClickHouse-User':'default','X-ClickHouse-Key':PW})
      r=local.c.getresponse(); data=r.read().decode()
      if r.status!=200: raise Exception(data[:150])
      a,b,c=data.strip().split('\t')
      return (ds,typ,rot,bits,dims,float(a),float(b),float(c),'')
    except Exception as e:
      try: local.c.close(); del local.c
      except: pass
      if att==3: return (ds,typ,rot,bits,dims,None,None,None,str(e)[:120])
      time.sleep(1.5)

results=[]; done=0; t0=time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
  for res in ex.map(run,configs):
    results.append(res); done+=1
    if done%100==0: print(f"{done}/{len(configs)} {time.time()-t0:.0f}s",flush=True)
with open('/tmp/recall_results.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['dataset','type','rotation','bits','dims','recall10','recall100','recall10in100'])
  for r in results: w.writerow(r[:8])
fails=[r for r in results if r[5] is None]
print(f"DONE {len(results)} configs in {time.time()-t0:.0f}s, failures={len(fails)}",flush=True)
for r in fails[:5]: print("FAIL",r,flush=True)
