# [Embeddings.info](https://embeddings.info/)

Datasets, pipelines, and an interactive explorer for large-scale image/text embeddings,
built on ClickHouse. Companion in spirit to [adsb.exposed](https://github.com/ClickHouse/adsb.exposed)
and [reversedns.space](https://github.com/ClickHouse/reversedns.space).

<img width="1325" height="971" alt="Screenshot_20260718_225129" src="https://github.com/user-attachments/assets/58de543d-aa71-41f5-872a-5d8202c5e420" />
<img width="1325" height="971" alt="Screenshot_20260718_212238" src="https://github.com/user-attachments/assets/a3eb925f-0596-47e8-9c9b-9b2a7fd477ad" />
<img width="1337" height="972" alt="Screenshot_20260718_030041" src="https://github.com/user-attachments/assets/134a2bd6-f8d4-4528-b8e3-2fa4ffd63a07" />

## Explorer website (`index.html`)

A zoomable, tile-rendered map of an embedding "sphere" — the whole corpus projected to 2-D via a
random Hadamard rotation, colored by density (OKLCH lightness) and a third projected axis (OKLCH hue).
Pure static HTML + Leaflet; it queries a ClickHouse HTTP endpoint directly (read-only `website` user).

- **Point-cloud mode** — each tile is a dense 1024x1024 `(r,g,b)` `RowBinary` raster (`ORDER BY pos WITH FILL`, zstd over HTTP) of pixels aggregated
  server-side (`GROUP BY` pixel, density→lightness, mean-z→hue), scattered into a canvas `ImageData`.
- **Thumbnail mode** — one representative image per 75×75 cell, rendered from `mmcommons.image_thumbs2`
  (75×75 gamma-correct RGB thumbnails) as PNG straight from ClickHouse.
- **Click → report** — nearest visible point → top-100 by `cosineDistance` over the embedding →
  thumbnail grid → click through to the full image.
- **Switchers** — dataset (`nomic` / `clip` / `siglip2`) and vector representation
  (original / Hadamard-rotated; bf16 / int8 / QBit bits+strides to follow).

Coordinates live in a `2^32 × 2^32` plane (`L.CRS.Simple`, `tile_size = 2^(32−z)`), exactly like the
adsb tile math. Tiles are fast thanks to a `mortonEncode(x,y)` projection.

## Data (ClickHouse service, `mmcommons` and `ann` databases)

- **`mmcommons.emb_{clip,nomic,siglip2}`** — ~99.1M YFCC100M image embeddings, `Array(BFloat16)`
  (768 / 768 / 1152-d), keyed by stripped MD5.
- **`mmcommons.image_thumbs` / `image_thumbs2`** — 99.1M gamma-correct 75×75 raw-RGB thumbnails
  (`v2` = center-square-crop + Lanczos-sharpened).
- **`mmcommons.image_blob`** — 99.6M source JPEG blobs; `yfcc_metadata` — Flickr metadata.
- **`ann.*`** — billion-scale ANN benchmark datasets (SIFT/DEEP/MSTuring/MSSpaceV/Text2Image 1B,
  LAION-400M, DINO-1B, MSMARCO, Wikipedia, OpenAI, Caselaw, YFCC, sparse SPLADE) with base + query +
  groundtruth. See the big-ann-benchmarks loaders in `scripts/`.

The projection columns and the read-only user are defined in [`sql/schema.sql`](sql/schema.sql).

## Scripts (`scripts/`)

Snapshot of the pipelines used to build and analyze the data (they assume a working dir and the
ClickHouse client in `$PATH`, with the write password in `$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD` — no
secrets are committed). Highlights:

- **Thumbnails** — `thumbs_s3*.py`, `thumb_s3*_chunk.sh`, `thumbs_s3*_run.sh` (streaming S3→resize→insert,
  gamma-correct in linear light, decoupled fetch/resize).
- **ANN datasets** — `binvec.py` (`.fbin/.u8bin/.i8bin`→RowBinary), `npyvec.py` (fp16 `.npy`→BFloat16),
  `gtparse.py` / `rangegt.py` (groundtruth), `sparse_load.py` (CSR), `annrun.sh` / `laion*.sh` runners.
- **Embeddings / analysis** — `embed*.py`, `quant*.py`, `hadamard_quant.py`, `train_{sentiment,topic}.py`,
  `*_recall*.sh`, `viz.py`, etc.

## Not included

Raw data, Parquet, the `clickhouse` binary, logs, and progress markers are excluded (see `.gitignore`).
