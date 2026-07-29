#!/usr/bin/env python3
"""Embed HackerNews comment `text` with a local sentence-transformers model and
stream the vectors to ClickHouse.

Source of truth is `hackernews_embeddings_qwen3_8b` (id, text) so every model
table lines up row-for-row with the Qwen3 table by `id`.

The id space is cut into fixed-width chunks; chunk c is handled by GPU (c % ngpu).
Each finished chunk is INSERTed into a per-model staging table
`stage_hn_<model>(id, vector Array(BFloat16))` with an insert_deduplication_token,
and a local `emb_text/<model>/<lo>.done` marker lets a restart skip it. Resumable
and idempotent. `finalize_text.sh` then builds the faithful QBit table.

Usage:
  CH_HOST=.. CH_PW=.. python embed_text.py --model minilm --gpu 0 --ngpu 4 [--batch 512] [--limit-chunks N]
"""
import argparse, io, os, sys, time
import numpy as np
import requests
import pyarrow as pa
import pyarrow.parquet as pq

from text_models import MODELS

CH_HOST = os.environ["CH_HOST"]
CH_USER = os.environ.get("CH_USER", "default")
CH_PW   = os.environ.get("CH_PW", os.environ.get("CH_PASSWORD", ""))
CH_PORT = os.environ.get("CH_PORT", "8443")
CH_URL  = f"https://{CH_HOST}:{CH_PORT}/"
CH_AUTH = (CH_USER, CH_PW)
SRC     = os.environ.get("SRC_TABLE", "hackernews_embeddings_qwen3_8b")

CHUNK   = int(os.environ.get("CHUNK", "200000"))   # id-width per chunk


def ch(query, data=None, params=None, timeout=900):
    p = {"query": query}
    if params:
        p.update(params)
    r = requests.post(CH_URL, params=p, data=data, auth=CH_AUTH, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"CH {r.status_code}: {r.text[:400]}")
    return r


def ensure_stage(model):
    ch(f"CREATE TABLE IF NOT EXISTS stage_hn_{model} "
       f"(id UInt32, vector Array(BFloat16)) ENGINE = MergeTree ORDER BY id")


def id_bounds():
    r = ch(f"SELECT min(id), max(id) FROM {SRC} FORMAT TSVRaw")
    lo, hi = r.text.split("\t")
    return int(lo), int(hi)


def read_chunk(lo, hi):
    """Return (ids: list[int], texts: list[str]) for id in [lo, hi).

    HN `text` is raw HTML with entities (e.g. &#34; , <a href=..>, <p>). Strip tags
    and decode entities server-side so the model embeds real prose, not markup.
    The row set is still defined by the raw `text != ''` filter (aligns with qwen3_8b)."""
    q = (f"SELECT id, decodeHTMLComponent(extractTextFromHTML(text)) "
         f"FROM {SRC} WHERE id >= {lo} AND id < {hi} AND text != '' "
         f"ORDER BY id FORMAT JSONCompactEachRow")
    r = ch(q, timeout=1200)
    ids, texts = [], []
    import json
    for line in r.iter_lines():
        if not line:
            continue
        row = json.loads(line)
        ids.append(row[0]); texts.append(row[1])
    return ids, texts


def push(model, lo, ids, vecs):
    if not ids:
        return
    buf = io.BytesIO()
    pq.write_table(pa.table({"id": pa.array(ids, type=pa.uint32()),
                             "vector": pa.array(list(vecs), type=pa.list_(pa.float32()))}), buf)
    q = (f"INSERT INTO stage_hn_{model} SELECT id, CAST(vector AS Array(BFloat16)) "
         f"FROM input('id UInt32, vector Array(Float32)') FORMAT Parquet")
    last = ""
    for attempt in range(6):
        r = requests.post(CH_URL, params={"query": q,
                                          "insert_deduplication_token": f"{model}_{lo}"},
                          data=buf.getvalue(), auth=CH_AUTH, timeout=1200)
        if r.status_code == 200:
            return
        last = f"{r.status_code} {r.text[:200]}"
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"push {model}/{lo} failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default=os.environ.get("OUT", os.path.join(os.getcwd(), "emb_text")))
    ap.add_argument("--limit-chunks", type=int, default=0, help="debug: process at most N chunks")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    donedir = os.path.join(args.out, args.model)
    os.makedirs(donedir, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer
    dev = f"cuda:{args.gpu}"
    print(f"[{args.model} gpu{args.gpu}] loading {cfg['id']} (dim {cfg['dim']}) on {dev} "
          f"-> stage_hn_{args.model} @ {CH_HOST}", flush=True)
    model = SentenceTransformer(cfg["id"], device=dev, trust_remote_code=cfg.get("trust", False),
                                model_kwargs={"torch_dtype": torch.bfloat16})
    if cfg.get("max_seq"):
        model.max_seq_length = cfg["max_seq"]
    prefix = cfg.get("prefix", "")

    if args.gpu == 0 or args.ngpu == 1:
        ensure_stage(args.model)
    else:
        # non-zero GPUs wait briefly so gpu0 creates the stage table first
        for _ in range(30):
            try:
                ch(f"EXISTS TABLE stage_hn_{args.model}")
                break
            except Exception:
                time.sleep(2)

    lo0, hi0 = id_bounds()
    chunks = [(lo, min(lo + CHUNK, hi0 + 1)) for lo in range(lo0, hi0 + 1, CHUNK)]
    mine = [(i, c) for i, c in enumerate(chunks) if i % args.ngpu == args.gpu]
    print(f"[{args.model} gpu{args.gpu}] {len(mine)}/{len(chunks)} chunks "
          f"(id {lo0}..{hi0}, width {CHUNK})", flush=True)

    done_n = 0
    t_start = time.time()
    rows_total = 0
    for ci, (i, (lo, hi)) in enumerate(mine):
        if args.limit_chunks and ci >= args.limit_chunks:
            break
        marker = os.path.join(donedir, f"{lo}.done")
        if os.path.exists(marker):
            done_n += 1
            continue
        ids, texts = read_chunk(lo, hi)
        if ids:
            enc_texts = [prefix + t for t in texts] if prefix else list(texts)
            if cfg.get("add_eos"):   # NV-Embed-v2 requires a trailing EOS on each input
                eos = model.tokenizer.eos_token or ""
                enc_texts = [t + eos for t in enc_texts]
            enc_kwargs = dict(batch_size=cfg.get("batch", args.batch), normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
            if cfg.get("task"):      # jina-v3 selects a LoRA adapter via the task kwarg
                enc_kwargs["task"] = cfg["task"]
            vecs = model.encode(enc_texts, **enc_kwargs).astype(np.float32)
            push(args.model, lo, ids, vecs)
            rows_total += len(ids)
        open(marker, "w").close()
        done_n += 1
        el = time.time() - t_start
        rate = rows_total / el if el > 0 else 0
        print(f"[{args.model} gpu{args.gpu}] chunk {lo}-{hi}: {len(ids)} rows "
              f"| {done_n}/{len(mine)} chunks | {rows_total} rows | {rate:.0f} rows/s", flush=True)

    print(f"[{args.model} gpu{args.gpu}] DONE {done_n} chunks, {rows_total} rows "
          f"in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
