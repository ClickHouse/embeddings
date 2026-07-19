# UMAP atlas stage

Computes 2D and 3D UMAP projections (each with an extra "color" dimension) of the nomic / clip / siglip2
embeddings and writes them back to ClickHouse, keyed by `md5`. GPU UMAP via RAPIDS cuML.

## Design decisions

- **Color = one extra UMAP dimension** → mapped to hue at viz time. So the 2D view is a **3-component**
  UMAP (`x, y` + `color`) and the 3D view a **4-component** UMAP (`x, y, z` + `color`). The `color` column
  is the raw (unbounded) UMAP axis; normalize to hue when plotting (see below).
- **Scale**: fit UMAP on a ~10M representative sample, then **transform all ~99.6M** points (cuML
  `transform` = project the rest by nearest neighbors). Every image gets coordinates.
- **PCA → 64d first**: 99.6M × 768–1152 dims (~300–460 GB fp32) won't fit RAM/GPU; PCA (fit on a sample,
  project all) reduces to 64d (~25 GB) so UMAP is tractable on a 24 GB L4.
- **Hardware**: cuML UMAP is single-GPU, so `run_umap.sh` (a) builds the 3 PCA reductions in parallel
  (one/GPU), then (b) runs the 6 `(model × {2d,3d})` UMAP jobs across all 4 GPUs, one job/GPU.
- **Restart-safe**: each `(model,dim)` transform truncates its target table first, so an interrupted run
  re-does exactly one clean pass (the PCA-reduced cache on disk persists — only the transform repeats).

## Tables (`umap/schema.sql`)

```
umap_<model>_2d (md5 String, x Float32, y Float32, color Float32)
umap_<model>_3d (md5 String, x Float32, y Float32, z Float32, color Float32)
```
for model ∈ {nomic, clip, siglip2}. Join to images/metadata by `md5`.

## Run

```bash
bash umap/setup_umap.sh                 # RAPIDS cuML + cupy + clickhouse-connect into isolated venv_umap
CH_HOST=... CH_PW=... ./umap/run_umap.sh
# validate first:  ./umap/umap_atlas.py --model nomic --limit 200000
```
`umap/autostart.sh` (optional) waits for the embedding job to finish, runs a small validation, and only
then launches the full atlas — so it can be armed to auto-continue unattended.

## Hue at viz time

`color` is a raw UMAP axis (unbounded, possibly skewed). Rank/quantile-normalize to `[0,1)` then ×360°:

```sql
SELECT md5, x, y, (color - q01) / (q99 - q01) AS hue01
FROM mmcommons.umap_nomic_2d,
     (SELECT quantile(0.01)(color) q01, quantile(0.99)(color) q99 FROM mmcommons.umap_nomic_2d)
```

## Tuning knobs (`umap_atlas.py`)

`--pca-dim 64` · `--fit-sample 10_000_000` · `--n-neighbors 30` · `--min-dist 0.1` · `--tf-batch 500_000`
· `--reduce-only` (build the PCA cache only). The long pole is transforming all 100M points; if cuML
`transform` is too slow at scale, the fallback is a pure kNN-average projection onto the fitted sample.
