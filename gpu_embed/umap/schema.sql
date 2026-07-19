-- UMAP atlas tables for mmcommons embeddings. Keyed by md5 (same key as emb_*/image_blob/yfcc_metadata,
-- so these join straight to the images + metadata). One table per (model, dimensionality).
--   *_2d: 2 position dims (x,y) + 1 color value   (from a 3-component UMAP; color = 3rd component)
--   *_3d: 3 position dims (x,y,z) + 1 color value (from a 4-component UMAP; color = 4th component)
-- color is a continuous UMAP axis (map to a colormap at viz time). Float32 is plenty for coordinates.

CREATE TABLE IF NOT EXISTS mmcommons.umap_nomic_2d   (md5 String, x Float32, y Float32, color Float32)              ENGINE = MergeTree ORDER BY md5;
CREATE TABLE IF NOT EXISTS mmcommons.umap_nomic_3d   (md5 String, x Float32, y Float32, z Float32, color Float32)   ENGINE = MergeTree ORDER BY md5;
CREATE TABLE IF NOT EXISTS mmcommons.umap_clip_2d    (md5 String, x Float32, y Float32, color Float32)              ENGINE = MergeTree ORDER BY md5;
CREATE TABLE IF NOT EXISTS mmcommons.umap_clip_3d    (md5 String, x Float32, y Float32, z Float32, color Float32)   ENGINE = MergeTree ORDER BY md5;
CREATE TABLE IF NOT EXISTS mmcommons.umap_siglip2_2d (md5 String, x Float32, y Float32, color Float32)              ENGINE = MergeTree ORDER BY md5;
CREATE TABLE IF NOT EXISTS mmcommons.umap_siglip2_3d (md5 String, x Float32, y Float32, z Float32, color Float32)   ENGINE = MergeTree ORDER BY md5;
