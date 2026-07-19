-- One table per model, keyed by the same stripped-md5 as image_blob / feat_* so they all join.
CREATE TABLE IF NOT EXISTS mmcommons.emb_siglip2  (md5 String, embedding Array(BFloat16) CODEC(ZSTD(1))) ENGINE=MergeTree ORDER BY md5;  -- google/siglip2-so400m-patch16-512, 1152-d
CREATE TABLE IF NOT EXISTS mmcommons.emb_clip     (md5 String, embedding Array(BFloat16) CODEC(ZSTD(1))) ENGINE=MergeTree ORDER BY md5;  -- open_clip ViT-L-14 (openai), 768-d
CREATE TABLE IF NOT EXISTS mmcommons.emb_nomic    (md5 String, embedding Array(BFloat16) CODEC(ZSTD(1))) ENGINE=MergeTree ORDER BY md5;  -- nomic-ai/nomic-embed-vision-v1.5, 768-d
CREATE TABLE IF NOT EXISTS mmcommons.emb_nemotron (md5 String, embedding Array(BFloat16) CODEC(ZSTD(1))) ENGINE=MergeTree ORDER BY md5;  -- nvidia/llama-nemotron-embed-vl-1b-v2, 2048-d
