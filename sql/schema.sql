-- Embeddings Explorer — ClickHouse schema for the projection + map + read-only access.
-- Applied to mmcommons.emb_siglip2 (dim 1152). The 768-dim datasets emb_clip / emb_nomic are
-- enriched the same way -- see the "768-dim datasets" section at the bottom for the exact DDL.
-- NOTE: randomHadamardTransform pads 1152 -> 2048 but leaves 768 -> 768 (it does NOT pad to the next
-- power of 2 here), so for clip/nomic the rotated QBit dim is 768 and x/y/z use 3 slices of 256.

-- 1) Hadamard-rotated embedding (stored, QBit) — basis for the projection and the "rotated"/bits+strides
--    representation switcher. randomHadamardTransform is deterministic; 1152 pads to 2048.
ALTER TABLE mmcommons.emb_siglip2
    ADD COLUMN embedding_rotated QBit(BFloat16, 2048, 128)
        DEFAULT randomHadamardTransform(CAST(embedding, 'Array(BFloat16)'));   -- needs allow_experimental_qbit_type=1

-- IMPORTANT: this column MUST be materialized (not left as a computed DEFAULT). The transposed distance
-- functions (cosineDistanceTransposed/...Quantized) cannot recompute a QBit DEFAULT on read — they read
-- the source column as empty and fail with Code 190 "must have size N, got 0" (ClickHouse#110634). A plain
-- MATERIALIZE COLUMN of x/y/z reads embedding_rotated on the fly but does NOT store it, so do this too:
ALTER TABLE mmcommons.emb_siglip2 MATERIALIZE COLUMN embedding_rotated
    SETTINGS mutations_sync = 0, allow_experimental_qbit_type = 1;

-- 1b) Additional QBit representations for the vector-representation switcher (strided BFloat16, and
--     Int8-quantized of both the original and the rotated). QBit dim = source vector length
--     (1152 for the original embedding, 2048 for the rotated). Int8 columns use CODEC(NONE).
--     IMPORTANT: the embeddings are L2-unit-normalized, so each coordinate is ~N(0, 1/sqrt(dim))
--     (std ~0.03). quantizeBFloat16ToInt8 is tanh-companding tuned for ~N(0,1) inputs, so we scale
--     each coordinate by sqrt(dim) before quantizing -> ~N(0,1) -> uses the full Int8 range (else the
--     Int8 values sit in a tiny +-3..50 band, wasting ~4 bits). sqrt(dim) = source vector length.
ALTER TABLE mmcommons.emb_siglip2
    ADD COLUMN `embedding_strided`     QBit(BFloat16, 1152, 128) DEFAULT CAST(embedding, 'Array(BFloat16)'),
    ADD COLUMN `embedding_int`         QBit(Int8, 1152, 128) DEFAULT arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt(1152))), CAST(embedding, 'Array(BFloat16)')) CODEC(NONE),
    ADD COLUMN `embedding_rotated_int` QBit(Int8, 2048, 128) DEFAULT arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt(2048))), CAST(embedding_rotated, 'Array(BFloat16)')) CODEC(NONE);
ALTER TABLE mmcommons.emb_siglip2
    MATERIALIZE COLUMN `embedding_strided`,
    MATERIALIZE COLUMN `embedding_int`,
    MATERIALIZE COLUMN `embedding_rotated_int`;

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

-- 4) Public read-only users for the browser (adsb.exposed model). CORS is enabled globally on the service.
--    Cloud disallows no_password, so a public (in-JS) password is used; security is the read-only grant + caps.
--    TWO users sharing the SAME password, so the many small thumbnail loads don't compete with (or count
--    against the limits of) the heavy analytical queries:
--      * website        - heavy queries: point-cloud/thumbnail tiles, top-100 similarity search, click lookups.
--      * website_thumbs - point queries: per-md5 thumbnail PNG loads (query cache on, few threads, tight caps).
CREATE USER IF NOT EXISTS website IDENTIFIED WITH sha256_password BY 'Embeddings-Viewer-2026!';
GRANT SELECT ON mmcommons.* TO website;
ALTER USER website SETTINGS
    allow_experimental_qbit_type = 1,
    use_query_cache = 1,
    query_cache_ttl = 864000,            -- 10 days (default 60s): layouts are immutable between re-materializations
    query_cache_nondeterministic_function_handling = 'save',   -- arrayJoin counts as non-deterministic -> error 704 otherwise
    max_execution_time = 60 MAX 300,
    max_memory_usage = 6000000000 MAX 12000000000,
    max_result_rows = 5000000 MAX 20000000,
    enable_parallel_replicas = 1,        -- fan heavy scans across all replicas of the `default` cluster
    automatic_parallel_replicas_mode = 0,-- 0 = don't let the cost heuristic opt out; always parallelize
    max_parallel_replicas = 999,         -- use every available replica (service has 3 -> verified 3/3 used)
    readonly = 2;

CREATE USER IF NOT EXISTS website_thumbs IDENTIFIED WITH sha256_password BY 'Embeddings-Viewer-2026!';
GRANT SELECT ON mmcommons.* TO website_thumbs;
ALTER USER website_thumbs SETTINGS
    use_query_cache = 1,                 -- thumbnails repeat a lot -> query cache is a big win
    query_cache_nondeterministic_function_handling = 'save',   -- screenshot render / hover tags use arrayJoin
    query_cache_ttl = 31536000,          -- 1 year (default is 60s; max safe is int32 seconds ~68y --
                                         -- larger values overflow internally to a PAST date, e.g. 100y -> 1990)
    max_execution_time = 10 MAX 30,
    max_memory_usage = 2000000000 MAX 4000000000,
    max_result_rows = 100000 MAX 1000000,
    max_threads = 2,                     -- point queries don't need parallelism; leave cores for `website`
    readonly = 2;

-- Tile query (point cloud), parameterized by {z,x,y,table}; returns a dense TILE*TILE x (r,g,b) UInt8 RowBinary raster (GROUP BY pos ORDER BY pos WITH FILL).
-- Density -> OKLCH lightness, mean projected-z -> OKLCH hue.  See site/index.html tileSQL().


-- ============================================================================================
-- 768-dim datasets: emb_clip, emb_nomic  (same columns/projection as siglip2, dims adjusted)
-- randomHadamardTransform does NOT pad 768 -> the rotated QBit is dim 768 (stride 128 -> 6 groups),
-- and x/y/z use 3 slices of 256 (arraySlice ...,1,256 / 257,256 / 513,256) covering all 768 dims.
-- The mean +- n*sigma normalization constants are measured PER MODEL (n_xy=2.5, n_z=0.7):
--   x: (sum + off_x)/scale_x,  off = n*sigma - mean,  scale = 2*n*sigma
--   clip : x +1.0025 /1.8655   y +0.8934 /1.72     z -0.2332 /0.5748
--   nomic: x +0.3414 /1.54     y +1.5506 /1.4005   z +0.527  /0.3644
-- Run scripts/enrich_clip_nomic.sh (3 passes, async mutations). Example for one table (clip):
--
-- ALTER TABLE mmcommons.emb_clip ADD COLUMN embedding_rotated QBit(BFloat16, 768, 128)
--     DEFAULT randomHadamardTransform(CAST(embedding, 'Array(BFloat16)'));
-- ALTER TABLE mmcommons.emb_clip MATERIALIZE COLUMN embedding_rotated;   -- must finish before x/y/z
-- ALTER TABLE mmcommons.emb_clip
--     ADD COLUMN embedding_strided     QBit(BFloat16, 768, 128) DEFAULT CAST(embedding, 'Array(BFloat16)'),
--     ADD COLUMN embedding_int         QBit(Int8, 768, 128) DEFAULT arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt(768))), CAST(embedding, 'Array(BFloat16)')) CODEC(NONE),
--     ADD COLUMN embedding_rotated_int QBit(Int8, 768, 128) DEFAULT arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt(768))), CAST(embedding_rotated, 'Array(BFloat16)')) CODEC(NONE),   -- rotated dim is also 768 (no padding)
--     ADD COLUMN x UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),  1,256)) + 1.0025)/1.8655, 0.,1.) * 4294967295)),
--     ADD COLUMN y UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),257,256)) + 0.8934)/1.72  , 0.,1.) * 4294967295)),
--     ADD COLUMN z UInt16 MATERIALIZED toUInt16(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),513,256)) - 0.2332)/0.5748, 0.,1.) * 65535));
-- ALTER TABLE mmcommons.emb_clip MATERIALIZE COLUMN embedding_strided, embedding_int, embedding_rotated_int, x, y, z;
-- ALTER TABLE mmcommons.emb_clip ADD PROJECTION proj_xy (SELECT x, y, z, md5 ORDER BY mortonEncode(x, y));
-- ALTER TABLE mmcommons.emb_clip MATERIALIZE PROJECTION proj_xy;
-- (emb_nomic identical with its own x/y/z constants above.)
