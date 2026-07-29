# GPU image-embedding pipeline (SigLIP2 · CLIP · nomic-vision · Nemotron-VL → ClickHouse)

Embeds the **~99M Multimedia Commons `data/images` JPEGs** (one medium-500 per photo) with **4 open models**,
reading straight from the public S3 bucket and writing 4 ClickHouse tables keyed by the same `md5`
(the stripped-MD5 hash), so they join to `yfcc_metadata`, `feat_*`, and `image_blob`.

| model | HF/id | dim | table |
|---|---|---|---|
| SigLIP2 | `google/siglip2-so400m-patch16-512` | 1152 | `mmcommons.emb_siglip2` |
| CLIP | open_clip `ViT-L-14` (openai) | 768 | `mmcommons.emb_clip` |
| nomic-vision | `nomic-ai/nomic-embed-vision-v1.5` | 768 | `mmcommons.emb_nomic` |
| Nemotron-VL | `nvidia/llama-nemotron-embed-vl-1b-v2` | 2048 | `mmcommons.emb_nemotron` |

## 1. Launch the instance
- **Type:** `g6.12xlarge` (4× L4, 48 vCPU, 192 GB) — best price/throughput. (`g6.48xlarge` = 8× L4 to halve wall-time; `g6e.12xlarge` = 4× L40S if Nemotron is slow.)
- **Region:** **us-west-2** (same as the bucket → S3 reads are free & fast).
- **AMI:** any GPU AMI **with an NVIDIA driver** — either the full **Deep Learning AMI** (torch preinstalled) or the lighter **Deep Learning Base OSS Nvidia Driver GPU AMI** (`setup.sh` installs torch itself). A bare non-GPU AMI won't work (no driver).
- **Storage:** the **root EBS is enough** — embeddings stream straight to ClickHouse, so nothing large is persisted; `OUT` (default `./emb`) holds only the HF model cache (~a few GB) and tiny per-prefix `.done` markers. Keep `OUT` on **EBS, not ephemeral NVMe**, so those markers survive a spot interruption and the job resumes.
- **Purchasing:** **Spot** (this is a resumable batch job) — ~70% cheaper.
- **IAM:** none needed for reading (bucket is public; the code uses unsigned S3). Only needed if you later stage output to your own S3.

## 2. Setup (once)
```bash
# copy this gpu_embed/ dir to the instance, then:
cd gpu_embed && bash setup.sh
```

## 3. Run — embeds AND uploads to ClickHouse as it goes (all GPUs, resumable)
```bash
export CH_HOST=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com CH_PW='<password>'
export OUT=/mnt/emb
./run.sh                     # all GPUs, all 4 models, batch 256
# monitor:  tail -f gpu0.log        progress:  ls /mnt/emb/siglip2/*.done | wc -l   # out of 4096
```
It fans the 4096 hash-prefixes across GPUs, reads images from S3 (48 decode threads), loads all 4 models on each GPU, embeds each batch with all four (S3 read + JPEG decode amortized across models), and **uploads each prefix's embeddings to `mmcommons.emb_*` the moment it's ready** (HTTP `INSERT … input() … CAST → Array(BFloat16)`). Tables are auto-created on start. Each prefix insert carries an `insert_deduplication_token` (`<model>_<prefix>`) so retries/resumes never double-insert; a local `/mnt/emb/<model>/<prefix>.done` marker lets a restart skip finished work. No large Parquet is kept on disk (only tiny `.done` markers).

## 4. (Optional) bulk-loader
`load_to_clickhouse.sh` is only needed for the **S3-staging alternative** (write Parquet → your bucket → `INSERT … SELECT s3()`) or to re-load from Parquet — the normal path in step 3 uploads directly.

## 5. Verify / use
```sql
SELECT count() FROM mmcommons.emb_siglip2;              -- ~99M each
-- join to metadata via md5:
SELECT m.title FROM mmcommons.emb_siglip2 e
  JOIN mmcommons.image_blob b USING (md5)               -- or feat_hybrid_cnn to reach photo_id
  LIMIT 10;
-- similarity search example:
SELECT md5, cosineDistance(embedding, (SELECT embedding FROM mmcommons.emb_siglip2 WHERE md5='<q>')) AS d
FROM mmcommons.emb_siglip2 ORDER BY d LIMIT 10;
```

## Cost / time (98–99M images)
- **SigLIP2 + CLIP + nomic** are light ViT encoders → the four-model pass is bottlenecked by S3 read + JPEG decode, ~a few k img/s aggregate on 4× L4. Wall-time ~**½–1.5 days**; compute ~**$50–200** on spot.
- **Nemotron-VL** (1B + dynamic tiling) is ~5–7× heavier and dominates — if included, expect ~**1–3 days** and **$300–800**. Consider running it as a separate pass (or on `g6e`/L40S).
- ClickHouse storage (**BFloat16**): siglip2 ~228 GB, nemotron ~406 GB, nomic/clip ~152 GB each ≈ **~0.94 TB** (half of Float32). Quantize further with your Int8/Lloyd-Max codec if needed.

## Caveats / knobs
- **Precision = BFloat16 everywhere.** Model weights + activations run in bf16; embeddings are stored as `Array(BFloat16)` (tables in `schema.sql`). The final L2-normalize is done in fp32 for unit-norm stability, and the intermediate Parquet stays fp32 (Apache Arrow has no bf16) — those files are transient on the box and cast to `BFloat16` at load time. If `CREATE TABLE … Array(BFloat16)` errors on your CH version, add `SETTINGS allow_experimental_bfloat16_type=1`.
- **Python ≥3.10 required for Nemotron-VL** (its remote code uses `X | Y` type unions). `setup.sh` auto-selects/installs a 3.10+ interpreter and builds the venv with it; the other 3 models also run on 3.9. If you're stuck on 3.9, run `./run.sh <ngpu> siglip2,clip,nomic` and do Nemotron separately.
- **Nemotron-VL loading is the one to verify first.** SigLIP2/CLIP/nomic use standard `transformers`/`open_clip` calls; Nemotron uses `trust_remote_code` and its image-encode API may differ — check the model card (build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2) and adjust `Encoders._nemotron` in `embed_images.py` (or run it via the NVIDIA **NIM** container). Test on one prefix first: `python3 embed_images.py --prefix-file <(echo 000) --out /tmp/t --models nemotron`.
- **Test each model on one prefix** before the full run (`--models siglip2` etc.) to confirm dims match `schema.sql`.
- **Throughput knobs:** `--batch` (256 → 512 if VRAM allows), decode threads (`max_workers=48` in the code), or add **NVIDIA DALI/nvJPEG** for GPU-side decode if CPU-bound.
- **All four models fit** on one 24 GB L4 together (~6–9 GB total); if VRAM is tight, run fewer models per pass (`--models siglip2,clip` then `--models nomic,nemotron`).
- Everything is keyed by `md5` = the filename hash = stripped-MD5(download_url) — the same key as the rest of `mmcommons`.
