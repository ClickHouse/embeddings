#!/usr/bin/env python3
"""Embed Multimedia Commons images with SigLIP2 / CLIP / nomic-embed-vision / Nemotron-VL.
Reads JPEGs straight from the public S3 bucket (anonymous), one 3-hex prefix at a time,
writes per-model Parquet (md5, embedding) to --out/<model>/<prefix>.parquet. Resumable.

Run one process per GPU (see run.sh), which sets CUDA_VISIBLE_DEVICES so this sees cuda:0.
"""
import os, io, sys, time, argparse, queue, threading, traceback
import numpy as np
from PIL import Image
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pyarrow as pa, pyarrow.parquet as pq
import requests
import torch, torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

BUCKET = "multimedia-commons"
ROOT   = "data/images"
DEV    = "cuda:0"
s3 = boto3.client("s3", region_name="us-west-2",
                  config=Config(signature_version=UNSIGNED, max_pool_connections=128,
                                retries={'max_attempts': 5, 'mode': 'standard'}))

DIMS = {"siglip2": 1152, "clip": 768, "nomic": 768, "nemotron": 2048}
NEM_SUBBATCH = int(os.environ.get("NEM_SUBBATCH", "8"))   # Nemotron tiles heavily -> tiny sub-batch avoids OOM

# ---- ClickHouse Cloud HTTP push (CH_URL/CH_AUTH set from env in main) ----
CH_URL = None; CH_AUTH = None
DDL = {m: f"CREATE TABLE IF NOT EXISTS mmcommons.emb_{m} (md5 String, embedding Array(BFloat16) CODEC(ZSTD(1))) ENGINE=MergeTree ORDER BY md5"
       for m in DIMS}

def ensure_tables(models):
    for m in models:
        requests.post(CH_URL, params={"query": DDL[m]}, auth=CH_AUTH, timeout=60).raise_for_status()

def push_ch(model, prefix, md5s, arr):
    """Upload one prefix's embeddings to ClickHouse now. Idempotent via insert_deduplication_token."""
    if not md5s: return
    buf = io.BytesIO()
    pq.write_table(pa.table({"md5": md5s, "embedding": pa.array(list(arr), type=pa.list_(pa.float32()))}), buf)
    data = buf.getvalue()
    # Name the columns explicitly: the table may have added DEFAULT/MATERIALIZED columns (e.g. QBit
    # quantized/Hadamard-rotated vectors, materialized x/y/z) — a column-list-less INSERT then fails with
    # "Number of columns doesn't match" because CH expects values for every insertable column.
    q = (f"INSERT INTO mmcommons.emb_{model} (md5, embedding) SELECT md5, CAST(embedding AS Array(BFloat16)) "
         f"FROM input('md5 String, embedding Array(Float32)') FORMAT Parquet")
    last = ""
    for attempt in range(5):
        r = requests.post(CH_URL, params={"query": q, "insert_deduplication_token": f"{model}_{prefix}"},
                          data=data, auth=CH_AUTH, timeout=900)
        if r.status_code == 200: return
        last = f"{r.status_code} {r.text[:200]}"; time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"push {model}/{prefix} failed: {last}")

# ----------------------------- encoders -----------------------------
class Encoders:
    def __init__(self, which):
        # Split each model into a CPU prep() half (decode-side, run in a thread pool) and a GPU fwd()
        # half (main thread). process_prefix pipelines them so preprocessing overlaps GPU compute.
        self.prep = {}; self.fwd = {}
        if "siglip2" in which:
            from transformers import AutoModel, AutoProcessor
            mid = "google/siglip2-so400m-patch16-512"
            self.sig_m = AutoModel.from_pretrained(mid, torch_dtype=torch.bfloat16).to(DEV).eval()
            self.sig_p = AutoProcessor.from_pretrained(mid)
            self.prep["siglip2"] = self._prep_siglip2; self.fwd["siglip2"] = self._fwd_siglip2
        if "clip" in which:
            import open_clip
            self.clip_m, _, self.clip_pp = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
            self.clip_m = self.clip_m.to(DEV).bfloat16().eval()
            self.prep["clip"] = self._prep_clip; self.fwd["clip"] = self._fwd_clip
        if "nomic" in which:
            from transformers import AutoModel, AutoImageProcessor
            mid = "nomic-ai/nomic-embed-vision-v1.5"
            # The published config.json ships `n_inner: 2048.0` (a float); transformers>=5 /
            # huggingface_hub>=1 strict-validate it as int and refuse to load. Coerce it in the
            # (already-downloaded) cached config so a fresh cache on any box also works.
            import json
            from huggingface_hub import hf_hub_download
            _cf = hf_hub_download(mid, "config.json")
            _d = json.load(open(_cf))
            if isinstance(_d.get("n_inner"), float):
                _d["n_inner"] = int(_d["n_inner"]); json.dump(_d, open(_cf, "w"))
            self.nom_m = AutoModel.from_pretrained(mid, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEV).eval()
            self.nom_p = AutoImageProcessor.from_pretrained(mid, use_fast=True)   # torchvision fast path
            self.prep["nomic"] = self._prep_nomic; self.fwd["nomic"] = self._fwd_nomic
        if "nemotron" in which:
            # VERIFY against the model card (build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2):
            # this model uses trust_remote_code; the exact image-encode call may differ (task/instruction).
            from transformers import AutoModel, AutoProcessor
            mid = "nvidia/llama-nemotron-embed-vl-1b-v2"
            # encode_documents() uses the model's own built-in self.processor, so no separate processor.
            self.nem_m = AutoModel.from_pretrained(mid, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEV).eval()
            self.prep["nemotron"] = self._prep_nemotron; self.fwd["nemotron"] = self._fwd_nemotron

    # Each model is a prep()/fwd() pair. prep() is pure CPU (image processor -> pinned bf16 CPU tensors)
    # and runs in the decode-side thread pool; fwd() is pure GPU (async H2D + model + fp32 L2-normalize)
    # and runs on the main thread. bf16 weights+activations; normalize in fp32 for unit-norm stability,
    # output fp32 (bf16-precision values) -> stored as Array(BFloat16).
    @staticmethod
    def _to_gpu(inp):   # dict of CPU tensors -> GPU, overlapping copy with prior compute
        return {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in inp.items()}

    def _prep_siglip2(self, imgs):
        inp = self.sig_p(images=imgs, return_tensors="pt")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16).pin_memory()
        return inp
    @torch.no_grad()
    def _fwd_siglip2(self, inp):
        f = self.sig_m.get_image_features(**self._to_gpu(inp))
        return F.normalize(f.float(), dim=-1).cpu().numpy()

    def _prep_clip(self, imgs):
        return torch.stack([self.clip_pp(im) for im in imgs]).to(torch.bfloat16).pin_memory()
    @torch.no_grad()
    def _fwd_clip(self, x):
        f = self.clip_m.encode_image(x.to(DEV, non_blocking=True))
        return F.normalize(f.float(), dim=-1).cpu().numpy()

    def _prep_nomic(self, imgs):
        inp = self.nom_p(imgs, return_tensors="pt")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16).pin_memory()
        return inp
    @torch.no_grad()
    def _fwd_nomic(self, inp):
        out = self.nom_m(**self._to_gpu(inp)).last_hidden_state[:, 0]   # CLS token
        return F.normalize(out.float(), dim=-1).cpu().numpy()

    # Nemotron's remote encode_documents() does its own (GPU) tiling+processing internally and is
    # GPU-bound, so there's no CPU prep to overlap: prep is a passthrough, all work is in fwd.
    # Dynamic tiling explodes memory (each image -> many 448px tiles), so sub-batch to avoid OOM.
    def _prep_nemotron(self, imgs):
        return list(imgs)
    @torch.no_grad()
    def _fwd_nemotron(self, imgs):
        outs = []
        for i in range(0, len(imgs), NEM_SUBBATCH):
            f = self.nem_m.encode_documents(images=imgs[i:i + NEM_SUBBATCH])
            outs.append(F.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, DIMS["nemotron"]), np.float32)

# ----------------------------- s3 io -----------------------------
def list_prefix(prefix):
    keys, token = [], None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=f"{ROOT}/{prefix}/")
        if token: kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            k = o["Key"]
            if k.endswith(".jpg"): keys.append(k)
        if not r.get("IsTruncated"): break
        token = r["NextContinuationToken"]
    return keys

def fetch(key):
    md5 = key.rsplit("/", 1)[1][:-4]          # filename minus .jpg == stripped md5
    try:
        b = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        im = Image.open(io.BytesIO(b)).convert("RGB")
        return md5, im
    except Exception:
        return None

# ----------------------------- per-prefix pipeline -----------------------------
def done_path(out, m, prefix): return f"{out}/{m}/{prefix}.done"

def process_prefix(prefix, enc, models, out, batch):
    # resumable: only (re)do models not yet pushed for this prefix
    need = [m for m in models if not os.path.exists(done_path(out, m, prefix))]
    if not need: return "skip", 0
    for m in need: os.makedirs(f"{out}/{m}", exist_ok=True)
    keys = list_prefix(prefix)
    if not keys:
        for m in need: open(done_path(out, m, prefix), "w").close()
        return "empty", 0
    md5s = {m: [] for m in need}
    vecs = {m: [] for m in need}
    DECODE = int(os.environ.get("DECODE_THREADS", "48"))
    PREP   = int(os.environ.get("PREP_THREADS", "8"))
    # Pipeline: decode (DECODE threads) -> CPU prep (PREP threads) -> GPU fwd (this thread). The bounded
    # queue of in-flight prep futures overlaps preprocessing with GPU compute (so the GPU stays fed) and
    # applies backpressure so we never hold more than ~queue-size batches in RAM.
    q = queue.Queue(maxsize=max(2, PREP + 2))
    def producer():
        buf_md5, buf_im = [], []
        try:
            with ThreadPoolExecutor(max_workers=DECODE) as dex, \
                 ThreadPoolExecutor(max_workers=PREP) as pex:
                def submit():
                    if not buf_im: return
                    mc, ic = buf_md5[:], buf_im[:]; buf_md5.clear(); buf_im.clear()
                    q.put((mc, pex.submit(lambda ims: {m: enc.prep[m](ims) for m in need}, ic)))
                for res in dex.map(fetch, keys):
                    if res is None: continue
                    buf_md5.append(res[0]); buf_im.append(res[1])
                    if len(buf_im) >= batch: submit()
                submit()
        finally:
            q.put(None)                       # sentinel (also on error, so the consumer can't hang)
    t = threading.Thread(target=producer, daemon=True); t.start()
    while True:
        item = q.get()
        if item is None: break
        mc, fut = item
        prepared = fut.result()               # parallel PREP workers keep this from blocking the GPU
        for m in need:
            vecs[m].append(enc.fwd[m](prepared[m])); md5s[m].extend(mc)
    t.join()
    n = 0
    for m in need:
        A = np.concatenate(vecs[m]).astype(np.float32) if vecs[m] else np.zeros((0, DIMS[m]), np.float32)
        push_ch(m, prefix, md5s[m], A)                 # <-- upload to ClickHouse as soon as this prefix is ready
        open(done_path(out, m, prefix), "w").close()   # mark pushed (resume skips it)
        n = len(md5s[m])
    return "ok", n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="siglip2,clip,nomic,nemotron")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    models = a.models.split(",")
    global CH_URL, CH_AUTH
    CH_URL = f"https://{os.environ['CH_HOST']}:8443/"
    CH_AUTH = (os.environ.get("CH_USER", "default"), os.environ["CH_PW"])
    ensure_tables(models)
    print(f"loading models {models} on {torch.cuda.get_device_name(0)}; pushing to {os.environ['CH_HOST']}", flush=True)
    enc = Encoders(models)
    prefixes = [l.strip() for l in open(a.prefix_file) if l.strip()]
    t0 = time.time(); done = 0; total = 0
    for p in prefixes:
        try:
            st, n = process_prefix(p, enc, models, a.out, a.batch)
        except Exception:
            print(f"FAIL {p}\n{traceback.format_exc()}", flush=True); continue
        done += 1; total += n
        if done % 5 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(prefixes)} prefixes, {total} imgs, {total/el:.0f} img/s, last={p}:{st}:{n}", flush=True)
    print(f"DONE {done} prefixes, {total} images, {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
