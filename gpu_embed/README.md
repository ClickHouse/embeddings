# GPU image-embedding pipeline (SigLIP2 · CLIP · nomic-vision · Nemotron-VL → ClickHouse)

Embeds the **~99.6M Multimedia Commons `data/images` JPEGs** (one medium-500 per photo) with **4 open
vision models**, reading straight from the public S3 bucket and writing ClickHouse tables keyed by the
same `md5` (the stripped-MD5 filename hash), so they join to `yfcc_metadata`, `feat_*`, and `image_blob`.
A second stage computes **2D+3D UMAP atlases** of those embeddings, also written back to ClickHouse.

| model | HF / id | dim | table | bottleneck |
|---|---|---|---|---|
| SigLIP2 | `google/siglip2-so400m-patch16-512` | 1152 | `mmcommons.emb_siglip2` | GPU-compute |
| CLIP | open_clip `ViT-L-14` (openai) | 768 | `mmcommons.emb_clip` | mixed |
| nomic-vision | `nomic-ai/nomic-embed-vision-v1.5` | 768 | `mmcommons.emb_nomic` | CPU-prep |
| Nemotron-VL | `nvidia/llama-nemotron-embed-vl-1b-v2` | 2048 | `mmcommons.emb_nemotron` | GPU-compute (heavy) |

Reference box: **`g6.12xlarge`** (4× NVIDIA L4, 48 vCPU, 192 GB) in **us-west-2** (same region as the
bucket → S3 reads free & fast). Everything is **resumable** (per-prefix `.done` markers) and streams
embeddings to ClickHouse as each of the 4096 hash-prefixes finishes — nothing large is persisted locally.

## Quick start

```bash
# 1. once: build the venv + install torch/transformers/open_clip/... (auto-picks Python >=3.10 for Nemotron)
bash setup.sh

# 2. embed. Cheapest model first, each a full parallel pass, auto-advancing (see run_seq.sh):
export CH_HOST=<clickhouse host> CH_PW='<password>'
./run_seq.sh 4                 # nomic -> clip -> siglip2, all 4 GPUs, resumable
#   or a single model:  ./run.sh 4 siglip2 256
#   Nemotron (5-7x heavier) is a separate pass, ideally on faster GPUs (g7/g6e):  ./run.sh 4 nemotron

# 3. UMAP atlas (2D+color and 3D+color per model) -> mmcommons.umap_<model>_{2d,3d}
bash umap/setup_umap.sh        # RAPIDS/cuML in an isolated venv_umap
CH_HOST=... CH_PW=... ./umap/run_umap.sh
```

Monitor: `tail -f gpu0.log` · progress `ls emb/siglip2/*.done | wc -l` (of 4096) · overall `run_seq.out`.

## How it works

- **`embed_images.py`** — per hash-prefix: lists S3, streams JPEGs (48 decode threads), and runs a
  **prefetch pipeline** (decode → parallel CPU preprocess → GPU forward) so preprocessing overlaps GPU
  compute. Each model is a `prep()`/`fwd()` pair. Embeddings are L2-normalized (fp32) and stored as
  `Array(BFloat16)`. Each prefix is uploaded the moment it's ready via `INSERT … (md5, embedding) SELECT …
  FROM input() FORMAT Parquet`, with an `insert_deduplication_token` so retries/resumes never double-insert.
- **`run.sh`** — fans the 4096 prefixes across GPU workers (1 process/GPU is the sweet spot — 2/GPU
  collapses throughput ~5× from CUDA-context contention). **Dynamic load balancing**: every worker gets the
  full prefix list rotated to a different start, and skips `.done` prefixes, so no GPU idles at the tail.
  **Auto-relaunch**: a supervisor restarts any worker that crashes/OOMs (resumes via `.done`).
- **`run_seq.sh`** — runs the light models **cheapest-first** (nomic < clip < siglip2), each a full pass,
  auto-advancing. Cheapest-first means the first usable table lands soonest at ~no extra cost.
- **`finish_siglip2.sh`** — partitions a run's remaining (contiguous-tail) prefixes into disjoint slices to
  avoid the rotation tail-collision on the last handful of prefixes (see docs/OPERATIONS.md).
- **`umap/`** — GPU UMAP atlas (PCA→64d → fit on a sample → transform all → ClickHouse). See docs/UMAP.md.

## Schema & precision

Tables in `schema.sql` (embeddings) and `umap/schema.sql` (atlases). Precision is **BFloat16** end-to-end
(weights + activations in bf16; final L2-normalize in fp32; stored `Array(BFloat16)`). ~0.94 TB total for
the 4 embedding tables. Everything keyed by `md5` = stripped-MD5(download_url).

## Docs

- **[docs/OPERATIONS.md](docs/OPERATIONS.md)** — runbook: monitoring, credentials, auto-relaunch, load
  balancing, measured throughput, cost/instance analysis, and every gotcha we hit + its fix.
- **[docs/UMAP.md](docs/UMAP.md)** — the UMAP atlas stage (design, params, schema, hue mapping).
