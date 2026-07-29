#!/usr/bin/env python3
"""Embed WonderfulWeb screenshots (nomic-vision / CLIP / SigLIP2) -> embedding service ClickHouse.

Source: WonderfulWeb cluster, table `web` (data JSON: screenshot Array(UInt8) 1280x1280x3, url, timestamp).
We take the LAST capture per url and downsample the screenshot to 512x512 SERVER-SIDE (memory-safe
arrayFilter with a constant keep-mask — a naive arrayMap gather replicates the 4.9MB array and OOMs),
so only ~786KB/image crosses the wire (raw would be ~93TB). One process per GPU; shard by url hash.

Writes (url, embedding) to <DEST> tables web_emb_img_<model>. Resumable via per-shard .done markers.

Env: WW_HOST, WW_PW (source);  CH_HOST, CH_PW, CH_USER (dest embedding service).
"""
import os, io, sys, time, argparse, traceback, urllib.request, urllib.parse, base64
import numpy as np
from PIL import Image
import pyarrow as pa, pyarrow.parquet as pq
import requests
import torch

sys.path.insert(0, os.path.dirname(__file__))
from embed_images import Encoders, DIMS   # reuse the mmcommons vision encoders (prep/fwd)

T = 512                      # downsample target (square); vision models resize internally
SS_DIM = 1280
OUTLEN = T * T * 3
WW_HOST = os.environ["WW_HOST"]; WW_PW = os.environ["WW_PW"]
WW_AUTH = "Basic " + base64.b64encode(f"default:{WW_PW}".encode()).decode()
CH_URL = f"https://{os.environ['CH_HOST']}:8443/"
CH_AUTH = (os.environ.get("CH_USER", "default"), os.environ["CH_PW"])

# server-side downsample: constant keep-mask (built once per query via WITH)
KA = f"arrayMap(r -> has(arrayMap(t->intDiv(t*{SS_DIM},{T}), range({T})), r), range({SS_DIM}))"
MASK = f"arrayMap(p -> ka[intDiv(intDiv(p,3),{SS_DIM})+1] AND ka[intDiv(p,3)%{SS_DIM}+1], range({SS_DIM*SS_DIM*3}))"


def ensure_tables(models):
    for m in models:
        q = (f"CREATE TABLE IF NOT EXISTS web_emb_img_{m} (url String, embedding Array(BFloat16) CODEC(ZSTD(1))) "
             f"ENGINE = SharedMergeTree('/clickhouse/tables/{{uuid}}/{{shard}}', '{{replica}}') ORDER BY url")
        requests.post(CH_URL, params={"query": q}, auth=CH_AUTH, timeout=60).raise_for_status()


def push_ch(model, shard, urls, arr):
    if not urls:
        return
    buf = io.BytesIO()
    pq.write_table(pa.table({"url": urls,
                             "embedding": pa.array(list(arr), type=pa.list_(pa.float32()))}), buf)
    q = (f"INSERT INTO web_emb_img_{model} (url, embedding) SELECT url, CAST(embedding AS Array(BFloat16)) "
         f"FROM input('url String, embedding Array(Float32)') FORMAT Parquet")
    last = ""
    for attempt in range(6):
        r = requests.post(CH_URL, params={"query": q, "insert_deduplication_token": f"{model}_{shard}"},
                          data=buf.getvalue(), auth=CH_AUTH, timeout=1200)
        if r.status_code == 200:
            return
        last = f"{r.status_code} {r.text[:200]}"; time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"push {model}/{shard} failed: {last}")


class RowBinaryReader:
    """Incremental RowBinary parser for rows of (url String, img Array(UInt8)) from a streaming response."""
    def __init__(self, resp, chunk=1 << 20):
        self.resp = resp; self.buf = b""; self.pos = 0; self.chunk = chunk; self.eof = False

    def _fill(self, need):
        while len(self.buf) - self.pos < need:
            d = self.resp.read(self.chunk)
            if not d:
                self.eof = True; return False
            if self.pos:                       # compact
                self.buf = self.buf[self.pos:]; self.pos = 0
            self.buf += d
        return True

    def _varint(self):
        shift = res = 0
        while True:
            if not self._fill(1):
                return None
            b = self.buf[self.pos]; self.pos += 1
            res |= (b & 0x7f) << shift
            if not (b & 0x80):
                return res
            shift += 7

    def read_row(self):
        n = self._varint()
        if n is None:
            return None
        if not self._fill(n):
            return None
        url = self.buf[self.pos:self.pos + n].decode("utf-8", "replace"); self.pos += n
        m = self._varint()
        if m is None or not self._fill(m):
            return None
        arr = np.frombuffer(self.buf[self.pos:self.pos + m], dtype=np.uint8); self.pos += m
        return url, arr


def stream_shard(lo, hi):
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    where = f"data.url >= '{esc(lo)}'" + (f" AND data.url < '{esc(hi)}'" if hi is not None else "")
    q = (f"WITH {KA} AS ka, {MASK} AS mask "
         f"SELECT url, arrayFilter((v,k)->k, ss, mask) AS img FROM ("
         f"  SELECT data.url AS url, data.screenshot AS ss FROM web "
         f"  WHERE {where} AND length(data.screenshot) = {SS_DIM*SS_DIM*3} "
         f"  ORDER BY data.url, data.timestamp DESC LIMIT 1 BY data.url) "
         f"SETTINGS max_memory_usage=25000000000, max_block_size=64, max_execution_time=0 FORMAT RowBinary")
    req = urllib.request.Request(f"https://{WW_HOST}:8443/", data=q.encode())
    req.add_header("Authorization", WW_AUTH)
    resp = urllib.request.urlopen(req, timeout=3600)
    rdr = RowBinaryReader(resp)
    while True:
        row = rdr.read_row()
        if row is None:
            break
        url, arr = row
        if arr.size != OUTLEN:               # skip malformed
            continue
        yield url, Image.fromarray(arr.reshape(T, T, 3))


def process_shard(idx, lo, hi, enc, models, out, batch):
    need = [m for m in models if not os.path.exists(f"{out}/{m}/{idx}.done")]
    if not need:
        return "skip", 0
    for m in need:
        os.makedirs(f"{out}/{m}", exist_ok=True)
    urls = {m: [] for m in need}; vecs = {m: [] for m in need}
    buf_u, buf_im = [], []

    def flush():
        if not buf_im:
            return
        prepared = {m: enc.prep[m](buf_im) for m in need}
        for m in need:
            vecs[m].append(enc.fwd[m](prepared[m])); urls[m].extend(buf_u)
        buf_u.clear(); buf_im.clear()

    n = 0
    for url, img in stream_shard(lo, hi):
        buf_u.append(url); buf_im.append(img)
        if len(buf_im) >= batch:
            flush()
    flush()
    for m in need:
        A = np.concatenate(vecs[m]).astype(np.float32) if vecs[m] else np.zeros((0, DIMS[m]), np.float32)
        push_ch(m, idx, urls[m], A)
        open(f"{out}/{m}/{idx}.done", "w").close()
        n = len(urls[m])
    return "ok", n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--bounds", default=os.path.join(os.path.dirname(__file__), "umap", "web_url_bounds.txt"))
    ap.add_argument("--models", default="siglip2,clip,nomic")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=os.environ.get("OUT", os.path.join(os.getcwd(), "emb_web_img")))
    ap.add_argument("--limit-shards", type=int, default=0)
    a = ap.parse_args()
    models = a.models.split(",")
    ensure_tables(models)
    print(f"[gpu{a.gpu}] loading {models} on {torch.cuda.get_device_name(0)} -> {os.environ['CH_HOST']}", flush=True)
    enc = Encoders(models)
    bounds = [l.rstrip("\n") for l in open(a.bounds) if l.strip()]
    shards = [(i, bounds[i], bounds[i + 1] if i + 1 < len(bounds) else None) for i in range(len(bounds))]
    mine = [s for s in shards if s[0] % a.ngpu == a.gpu]
    print(f"[gpu{a.gpu}] {len(mine)}/{len(shards)} shards", flush=True)
    t0 = time.time(); done = 0; total = 0
    for ci, (idx, lo, hi) in enumerate(mine):
        if a.limit_shards and ci >= a.limit_shards:
            break
        try:
            st, k = process_shard(idx, lo, hi, enc, models, a.out, a.batch)
        except Exception:
            print(f"[gpu{a.gpu}] FAIL shard {idx}\n{traceback.format_exc()}", flush=True); time.sleep(5); continue
        done += 1; total += k
        el = time.time() - t0
        print(f"[gpu{a.gpu}] shard {idx} [{lo[:30]}..]: {st} {k} | {done}/{len(mine)} | {total} imgs | {total/el:.0f} img/s", flush=True)
    print(f"[gpu{a.gpu}] DONE {done} shards, {total} imgs in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
