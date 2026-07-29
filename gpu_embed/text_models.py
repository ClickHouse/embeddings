"""Registry of local text-embedding models, ordered smallest -> largest.

Each entry:
  key            short name -> ClickHouse table  hackernews_embeddings_<key>
  id             HuggingFace / sentence-transformers model id
  dim            embedding dimension
  prefix         string prepended to every document before encoding
                 (models like E5 / nomic require a "passage:" / "search_document:" prefix)
  trust          pass trust_remote_code=True to sentence-transformers
  max_seq        optional override of max_seq_length (tokens)
  gated          needs HF_TOKEN (accept license on the hub first)
"""

MODELS = {
    # ---- 384-dim, ~20-35M params: the smallest, run these first ------------------
    "minilm": dict(id="sentence-transformers/all-MiniLM-L6-v2", dim=384, prefix="",
                   trust=False, max_seq=256),
    "arctic_xs": dict(id="Snowflake/snowflake-arctic-embed-xs", dim=384, prefix="",
                      trust=False, max_seq=512),
    "bge_small": dict(id="BAAI/bge-small-en-v1.5", dim=384, prefix="",
                      trust=False, max_seq=512),
    "gte_small": dict(id="thenlper/gte-small", dim=384, prefix="",
                      trust=False, max_seq=512),
    "e5_small": dict(id="intfloat/e5-small-v2", dim=384, prefix="passage: ",
                     trust=False, max_seq=512),
    "granite_small": dict(id="ibm-granite/granite-embedding-30m-english", dim=384, prefix="",
                          trust=False, max_seq=512),

    # ---- 512 / 768-dim, ~100-300M params ---------------------------------------
    "jina_small": dict(id="jinaai/jina-embeddings-v2-small-en", dim=512, prefix="",
                       trust=True, max_seq=512),
    "arctic_m": dict(id="Snowflake/snowflake-arctic-embed-m", dim=768, prefix="",
                     trust=False, max_seq=512),
    "bge_base": dict(id="BAAI/bge-base-en-v1.5", dim=768, prefix="",
                     trust=False, max_seq=512),
    "gte_base": dict(id="thenlper/gte-base", dim=768, prefix="",
                     trust=False, max_seq=512),
    "e5_base": dict(id="intfloat/e5-base-v2", dim=768, prefix="passage: ",
                    trust=False, max_seq=512),
    # max_seq 512 (not 8192) + small batch: batch 512 x seq 2048 OOMs an L4 (24GB).
    "nomic": dict(id="nomic-ai/nomic-embed-text-v1.5", dim=768, prefix="search_document: ",
                  trust=True, max_seq=512, batch=128),
    "embeddinggemma": dict(id="google/embeddinggemma-300m", dim=768,
                           prefix="title: none | text: ", trust=False, max_seq=512, gated=True, batch=128),

    # ---- 896 / 1024-dim, larger (smaller batches to stay within L4 VRAM) --------
    "kalm": dict(id="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5", dim=896,
                 prefix="", trust=True, max_seq=512, batch=32),   # ~0.5B params
    "bge_large": dict(id="BAAI/bge-large-en-v1.5", dim=1024, prefix="",
                      trust=False, max_seq=512, batch=256),
    "e5_large": dict(id="intfloat/e5-large-v2", dim=1024, prefix="passage: ",
                     trust=False, max_seq=512, batch=256),
    "jina_v3": dict(id="jinaai/jina-embeddings-v3", dim=1024, prefix="",
                    trust=True, max_seq=512, task="retrieval.passage", batch=32),

    # ---- heavy, run last -------------------------------------------------------
    # NV-Embed-v2 (7B, ~15.7GB bf16): needs trailing EOS; batch 8 OOMs an L4 (24GB), so batch=2
    # + expandable_segments (set in run_text.sh). Very slow (~days for 37.8M) — the final model.
    "nemotron": dict(id="nvidia/NV-Embed-v2", dim=4096, prefix="",
                     trust=True, max_seq=512, add_eos=True, batch=2),
}
