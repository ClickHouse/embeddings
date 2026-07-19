# Operations runbook

Hard-won operational notes for running the embedding + UMAP pipeline at ~99.6M-image scale on 4× L4.

## Credentials

`run*.sh` need `CH_HOST` and `CH_PW` in the environment (`run.sh` aborts at its `${CH_HOST:?}` guard
without them). A fresh shell may not have them; the launching user exported them interactively.
Verify with `curl "https://$CH_HOST:8443/" --user default:$CH_PW --data-binary "SELECT 1"` → `1`.
**Do not kill all workers before you can relaunch** — a running worker's env holds the only easily
recoverable copy of `CH_PW` (`tr '\0' '\n' < /proc/<pid>/environ`).

## Monitoring

- Progress: `ls emb/<model>/*.done | wc -l` (of 4096); rows: `SELECT count() FROM mmcommons.emb_<model>`.
- Per-worker rate: workers log `N/4096 prefixes, M imgs, R img/s` every 5 prefixes to `gpu<w>.log`.
- **Throughput via ClickHouse row-count needs a LONG window** (≥300s): inserts are bursty (one per prefix,
  and a big prefix is ~27k rows every few minutes) — short windows read 0 and mislead.
- GPU: `nvidia-smi dmon` (scrolling) or `nvtop`. Note L4 util is *bursty* for the CPU-prep-bound light
  models (nomic/clip) — that's normal, not a stall.
- **Watch out for grep matching your own command line** (`pgrep -f embed_images.py`, `ps|grep umap_atlas`
  in a command that contains those strings will match the shell running it) — it caused false "extra
  process" readings. Use `[e]mbed_images` bracket tricks or check `nvidia-smi --query-compute-apps`.

## Load balancing & the tail

`run.sh` gives each worker the **full** prefix list rotated to a different start, relying on the `.done`
skip. This avoids idle GPUs mid-run (a static `NR%N` split leaves fast GPUs idle for hours because prefix
sizes vary 5–27k images). **But at the contiguous tail** (e.g. the last ~84 prefixes `fac..fff`, all
~27k), all workers converge and grind the *same* prefix in lockstep → ~4× wasted compute, ~1 unique
prefix done per ~15 min. Fix: **`finish_siglip2.sh`** partitions the remaining undone prefixes into
disjoint slices (one per GPU) → true 4× parallelism on the tail. (A durable fix would be atomic per-prefix
claiming; partitioning the tail is the pragmatic one.)

## Auto-relaunch (supervisor)

`run.sh` supervises each worker: on non-zero exit (crash / kernel OOM-kill — silent, no Python traceback)
it relaunches, resuming via `.done`. **Critical:** the supervisor sets `set +e` — `run.sh` runs under
`set -e`, which would otherwise abort the supervisor on the worker's failure *before* it could relaunch.
Verified by killing a worker and watching it come back.

## Measured throughput (4× L4, g6.12xlarge)

| model | aggregate | notes |
|---|---|---|
| nomic | ~1,070 img/s | CPU-prep-bound; prefetch pipeline ~2× over naive; 1 proc/GPU |
| clip | ~640 img/s | ViT-L, heavier |
| siglip2 | ~300 img/s | so400m @512px (1024 patches), GPU-compute-bound |
| Nemotron | ~10–100 img/s (est.) | 1B + dynamic tiling; run separately on faster GPUs |

Full 4-model pass on 4× L4 is impractically long (~months with Nemotron) — that's why the light 3 run
here and **Nemotron belongs on g7 (Blackwell) / g6e (L40S)**. See the cost analysis below.

### Cost / instance notes
- On $/TFLOP, AWS on-demand is ~2–3× pricier than GPU-first clouds for the same silicon.
- **G7 (RTX PRO Blackwell) on spot** was dramatically cheap at time of writing ($1.21–4.39/hr for 2–8
  GPUs) — best cost/perf by far; verify with `aws ec2 describe-spot-price-history` (spot is volatile).
- For text-embedding serving (e.g. Qwen3-Embedding-8B): ~$0.02–0.03/1M tokens on cheap neocloud GPUs,
  ~half that in FP8; a serverless embedding API (~$0.01/1M) often beats self-hosting on-demand.

## Gotchas we hit (and the fixes, all in the code)

1. **transformers 5.x breaks the trust_remote_code models** (nomic-vision, Nemotron were written for the
   4.x API; v5 renamed internals like `_tied_weights_keys`). `setup.sh` pins `transformers>=4.49,<5`.
2. **nomic config `n_inner: 2048.0`** (a float) is rejected by strict-dataclass validation in newer libs.
   `embed_images.py` coerces it to int in the cached `config.json` before load.
3. **Nemotron-VL API**: it's a retrieval VL model — embed images via `model.encode_documents(images=…)`
   (which runs its own processor + pooling), not `get_image_features`.
4. **Nemotron OOM**: dynamic tiling explodes memory; sub-batch it (`NEM_SUBBATCH=8`) so the shared batch
   of 256 doesn't OOM a 24 GB L4. Also `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
5. **1 proc/GPU, not 2**: two CUDA contexts on one L4 (no MPS) serialize kernels → throughput collapses
   ~5× (measured 351→65 img/s). Keep `PROCS_PER_GPU=1`.
6. **QBit / altered target table**: if `emb_*` gains `DEFAULT`/`MATERIALIZED` columns (e.g. ClickHouse
   `QBit` quantized / `randomHadamardTransform` / materialized `x/y/z`), a column-list-less
   `INSERT … SELECT` fails with *"Number of columns doesn't match (source N, result M)"*. Fix: name the
   columns — `INSERT INTO emb_<m> (md5, embedding) SELECT …` — so DEFAULT/MATERIALIZED auto-compute.
   (The QBit `DEFAULT` columns are computed per-insert on the server ~+2× insert time, but that stays
   cheap ~8s/27k-prefix — not the throughput bottleneck.)
7. **`use_fast=True`** on the nomic image processor (torchvision fast path) — silences the "slow processor"
   warning and speeds CPU-side preprocessing.

## UMAP-stage gotchas
8. **PCA fit-sample OOM on high-dim models**: siglip2 is 1152-dim, so a 3M PCA fit-sample = 13.8 GB and
   cuML's ~2× working copy OOMs a 24 GB L4 (768-dim nomic/clip squeak by). `umap_atlas.py` caps the
   fit-sample by a memory budget (`PCA_FIT_BYTES`, default 6 GB) — ~1M rows is plenty for 64 PCs.
9. **Concurrent reduce race**: if the phase-1 `--reduce-only` fails, phase-2's two jobs for that model
   (2d & 3d) both rebuild the reduce and race on the shared `red/md5` files. Data is deterministic so it's
   usually benign, but it's now guarded by a build lock (only one process builds; others wait for cache).
10. **Tail idle at the end of UMAP**: with 6 jobs on 4 GPUs, the last model (siglip2 2d+3d) leaves 2 GPUs
    idle. To use them, split each transform's point-range across GPUs (not implemented; single-GPU per job).
