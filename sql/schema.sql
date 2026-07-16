-- Embeddings Explorer — ClickHouse schema for the projection + map + read-only access.
-- Applied to mmcommons.emb_siglip2 (dim 1152); replicate for emb_clip / emb_nomic (dim 768:
-- use 3 slices of 256 => arraySlice(v,1,256),(257,256),(513,256), and recompute the n*sigma constants).

-- 1) Hadamard-rotated embedding (stored, QBit) — basis for the projection and the "rotated"/bits+strides
--    representation switcher. randomHadamardTransform is deterministic; 1152 pads to 2048.
ALTER TABLE mmcommons.emb_siglip2
    ADD COLUMN embedding_rotated QBit(BFloat16, 2048, 128)
        DEFAULT randomHadamardTransform(CAST(embedding, 'Array(BFloat16)'));   -- needs allow_experimental_qbit_type=1

-- 2) 2-D projection + hue as MATERIALIZED columns. Range = mean ± n*sigma (precomputed constants,
--    n_xy = 2.5 ≈ q01/q99, n_z = 0.7 ≈ q25/q75) so no window functions are needed at read time.
--    siglip2 constants: xf μ -0.116 σ 0.291 ; yf μ 0.048 σ 0.299 ; zf μ 0.011 σ 0.231.
ALTER TABLE mmcommons.emb_siglip2
    ADD COLUMN x UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),  1,384)) + 0.8439)/1.456 , 0., 1.) * 4294967295)),
    ADD COLUMN y UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),385,384)) + 0.6995)/1.495 , 0., 1.) * 4294967295)),
    ADD COLUMN z UInt16 MATERIALIZED toUInt16(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),769,384)) + 0.1506)/0.3234, 0., 1.) * 65535));

ALTER TABLE mmcommons.emb_siglip2 MATERIALIZE COLUMN x, MATERIALIZE COLUMN y, MATERIALIZE COLUMN z
    SETTINGS mutations_sync = 0, allow_experimental_qbit_type = 1;

-- 3) Projection sorted by the Z-order curve over (x,y) so 2-D tile range queries prune granules
--    (space-filling-curve index analysis). MUST include md5: the point-cloud tile aggregation uses
--    x,y,z, but the nearest-point-for-report and thumbnail-per-cell lookups SELECT md5 while filtering
--    x,y -- without md5 in the projection those fall back to the md5-ordered main table (full scan).
--    With md5 in the projection they prune via morton (verified: 3 MiB / 355K rows vs full scan).
ALTER TABLE mmcommons.emb_siglip2 ADD PROJECTION proj_xy (SELECT x, y, z, md5 ORDER BY mortonEncode(x, y));
ALTER TABLE mmcommons.emb_siglip2 MATERIALIZE PROJECTION proj_xy
    SETTINGS mutations_sync = 0, allow_experimental_qbit_type = 1;

-- 4) Public read-only user for the browser (adsb.exposed model). CORS is enabled globally on the service.
--    Cloud disallows no_password, so a public (in-JS) password is used; security is the read-only grant + caps.
CREATE USER IF NOT EXISTS website IDENTIFIED WITH sha256_password BY 'Embeddings-Viewer-2026!';
GRANT SELECT ON mmcommons.* TO website;
ALTER USER website SETTINGS
    allow_experimental_qbit_type = 1,
    use_query_cache = 1,
    max_execution_time = 60 MAX 300,
    max_memory_usage = 6000000000 MAX 12000000000,
    max_result_rows = 5000000 MAX 20000000,
    readonly = 2;

-- Tile query (point cloud), parameterized by {z,x,y,table}; returns sparse (px,py,r,g,b) RowBinary.
-- Density -> OKLCH lightness, mean projected-z -> OKLCH hue.  See site/index.html tileSQL().
