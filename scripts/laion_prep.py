import concurrent.futures as cf, urllib.request, ast
BASE="https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings"
def count(n):
    req=urllib.request.Request(f"{BASE}/img_emb/img_emb_{n}.npy", headers={'Range':'bytes=0-149'})
    d=urllib.request.urlopen(req,timeout=40).read()
    hl=int.from_bytes(d[8:10],'little'); h=ast.literal_eval(d[10:10+hl].decode('latin1').strip())
    return n, h['shape'][0]
res={}
with cf.ThreadPoolExecutor(32) as ex:
    for n,c in ex.map(count, range(410)): res[n]=c
off=0; out=[]
for n in range(410):
    out.append(f"{n} {off} {res[n]}"); off+=res[n]
open('/tmp/laion_shards.txt','w').write("\n".join(out)+"\n")
print("shards=410 total_vectors=",off)
