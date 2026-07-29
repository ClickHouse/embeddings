#!/usr/bin/env python3
"""Embed WonderfulWeb page text (title + dom.textContent) with the large HN text models + nomic ->
embedding service ClickHouse.

Last capture per url; url-range shards (PK = data.url) read from a boundaries file. All models loaded
per GPU (text read is cheap, so amortize it). Writes (url, embedding) to web_emb_txt_<model>.

Env: WW_HOST, WW_PW (source);  CH_HOST, CH_PW, CH_USER (dest).
"""
import os, io, sys, time, argparse, traceback, urllib.request, base64, json
import numpy as np
import requests
import pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from text_models import MODELS

WEB_MODELS = ["bge_large", "e5_large", "jina_v3", "nomic"]
WW_HOST = os.environ["WW_HOST"]; WW_PW = os.environ["WW_PW"]
WW_AUTH = "Basic " + base64.b64encode(f"default:{WW_PW}".encode()).decode()
CH_URL = f"https://{os.environ['CH_HOST']}:8443/"
CH_AUTH = (os.environ.get("CH_USER", "default"), os.environ["CH_PW"])
MAXTC = int(os.environ.get("MAXTC", "8000"))   # cap textContent chars before the model truncates anyway


def ensure_tables(models):
    for m in models:
        q = (f"CREATE TABLE IF NOT EXISTS web_emb_txt_{m} (url String, embedding Array(BFloat16) CODEC(ZSTD(1))) "
             f"ENGINE = SharedMergeTree('/clickhouse/tables/{{uuid}}/{{shard}}', '{{replica}}') ORDER BY url")
        requests.post(CH_URL, params={"query": q}, auth=CH_AUTH, timeout=60).raise_for_status()


def push_ch(model, shard, urls, arr):
    if not urls:
        return
    buf = io.BytesIO()
    pq.write_table(pa.table({"url": urls, "embedding": pa.array(list(arr), type=pa.list_(pa.float32()))}), buf)
    q = (f"INSERT INTO web_emb_txt_{model} (url, embedding) SELECT url, CAST(embedding AS Array(BFloat16)) "
         f"FROM input('url String, embedding Array(Float32)') FORMAT Parquet")
    last = ""
    for attempt in range(6):
        r = requests.post(CH_URL, params={"query": q, "insert_deduplication_token": f"{model}_{shard}"},
                          data=buf.getvalue(), auth=CH_AUTH, timeout=1200)
        if r.status_code == 200:
            return
        last = f"{r.status_code} {r.text[:200]}"; time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"push {model}/{shard} failed: {last}")


def read_shard(lo, hi):
    """(url, text) for last capture per url in [lo, hi). text = title + textContent."""
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    where = f"data.url >= '{esc(lo)}'" + (f" AND data.url < '{esc(hi)}'" if hi is not None else "")
    q = ("SELECT data.url AS url, concat(data.dom.title::String, '\\n', "
         f"substr(data.dom.textContent::String, 1, {MAXTC})) AS text FROM web "
         f"WHERE {where} ORDER BY data.url, data.timestamp DESC LIMIT 1 BY data.url "
         "FORMAT JSONCompactEachRow SETTINGS max_execution_time=0")
    req = urllib.request.Request(f"https://{WW_HOST}:8443/", data=q.encode()); req.add_header("Authorization", WW_AUTH)
    raw = urllib.request.urlopen(req, timeout=1800).read()
    urls, texts = [], []
    # split on \n ONLY: JSONCompactEachRow separates rows by \n; str.splitlines() also breaks on
    # unicode separators (U+2028, U+0085/NEL, ...) that appear raw inside textContent JSON strings,
    # cutting a row mid-string -> "Unterminated string" JSONDecodeError (deterministic per poison shard).
    for line in raw.decode("utf-8", "replace").split("\n"):   # some textContent has invalid utf-8 bytes
        if not line.strip():
            continue
        u, t = json.loads(line)
        urls.append(u); texts.append(t or "")
    return urls, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--bounds", default=os.path.join(os.path.dirname(__file__), "umap", "web_url_bounds.txt"))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=os.environ.get("OUT", os.path.join(os.getcwd(), "emb_web_txt")))
    ap.add_argument("--limit-shards", type=int, default=0)
    a = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda:0"   # GPU is selected via CUDA_VISIBLE_DEVICES; --gpu is only for shard assignment
    ensure_tables(WEB_MODELS)
    models = {}
    for m in WEB_MODELS:
        cfg = MODELS[m]
        sm = SentenceTransformer(cfg["id"], device=dev, trust_remote_code=cfg.get("trust", False),
                                 model_kwargs={"torch_dtype": torch.bfloat16})
        if cfg.get("max_seq"):
            sm.max_seq_length = cfg["max_seq"]
        models[m] = (sm, cfg)
    print(f"[gpu{a.gpu}] loaded {WEB_MODELS} on {torch.cuda.get_device_name(0)}", flush=True)

    bounds = [l.rstrip("\n") for l in open(a.bounds) if l.strip()]
    shards = [(bounds[i], bounds[i + 1] if i + 1 < len(bounds) else None) for i in range(len(bounds))]
    mine = [(i, s) for i, s in enumerate(shards) if i % a.ngpu == a.gpu]
    print(f"[gpu{a.gpu}] {len(mine)}/{len(shards)} shards", flush=True)
    for m in WEB_MODELS:
        os.makedirs(f"{a.out}/{m}", exist_ok=True)

    t0 = time.time(); done = 0; total = 0
    for ci, (i, (lo, hi)) in enumerate(mine):
        if a.limit_shards and ci >= a.limit_shards:
            break
        need = [m for m in WEB_MODELS if not os.path.exists(f"{a.out}/{m}/{i}.done")]
        if not need:
            done += 1; continue
        try:
            urls, texts = read_shard(lo, hi)
            if urls:
                for m in need:
                    sm, cfg = models[m]
                    pfx = cfg.get("prefix", "")
                    enc_texts = [pfx + t for t in texts] if pfx else texts
                    kw = dict(batch_size=a.batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                    if cfg.get("task"):
                        kw["task"] = cfg["task"]
                    vecs = sm.encode(enc_texts, **kw).astype(np.float32)
                    push_ch(m, i, urls, vecs)
                    open(f"{a.out}/{m}/{i}.done", "w").close()
            else:
                for m in need:
                    open(f"{a.out}/{m}/{i}.done", "w").close()
        except Exception:
            print(f"[gpu{a.gpu}] FAIL shard {i}\n{traceback.format_exc()}", flush=True); time.sleep(5); continue
        done += 1; total += len(urls) if urls else 0
        el = time.time() - t0
        print(f"[gpu{a.gpu}] shard {i} [{lo[:30]}..]: {len(urls) if urls else 0} | {done}/{len(mine)} | {total} | {total/el:.0f} txt/s", flush=True)
    print(f"[gpu{a.gpu}] DONE {done} shards, {total} texts in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
