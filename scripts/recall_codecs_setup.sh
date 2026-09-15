#!/usr/bin/env bash
# Build the bench.* tables that scripts/recall_codecs_bench.py consumes, for the 7 models that have
# Quantized-codec sibling tables (see CODEC_MODELS in index.html).
#
# Two kinds of table:
#   bench.samp_<label>         : ~100k-row pool. Photos reuses the samp_emb_* tables recall_bench_setup.sh
#                                built (rebuilding would move the sample and invalidate results/recall.csv);
#                                web/hn pools are created here.
#   bench.cs_<label>_<codec>   : the same pool again, with `embedding` carrying CODEC(Quantized(...)).
#                                Needed because that codec is CREATE TABLE-only -- it cannot be added to an
#                                existing table by ALTER, so the codec pool has to be its own table.
#   bench.qc_/gtc_<label>      : NQ query points + their exact top-100, computed FROM THE CODEC POOL with the
#                                identical expression the benchmark's search uses -- cosineDistance over two
#                                Array(BFloat16) operands. The older gt_* tables cast to Array(Float32) first,
#                                and that precision gap let near-ties rank differently between ground truth and
#                                search: a control run with the codec disabled scored 0.980 recall@10 on
#                                web/img_siglip2 where it must be exactly 1.000. Deriving gt here in the search's
#                                own precision makes the control exact by construction, at the cost of no longer
#                                sharing a ground truth with recall.html (which measures a different thing).
#
# Every table keys on `md5 String` whatever the corpus (web -> url, hn -> toString(id)) so the harness and
# the ground-truth arrays stay uniform.
#
# Sampling differs by corpus on purpose: md5 is itself a hash, so photos' "first 100k by md5" is already
# random w.r.t. the embeddings. `url` and `id` are NOT -- ordering by them would take the alphabetically
# first domains / the oldest comments. Those use cityHash64(key) % stride instead.
#
# Usage:  CH_CMD=/path/to/client-wrapper ./scripts/recall_codecs_setup.sh
#   where the wrapper is an executable that runs `clickhouse client` with host/user/password already set
#   and forwards "$@" (keeps credentials out of the repo).
set -e
NQ="${NQ:-100}"          # query points per pool
CH_CMD="${CH_CMD:-clickhouse}"
OPTS=(--allow_experimental_qbit_type 1 --enable_quantized_codec 1 --max_execution_time 900
      --enable_parallel_replicas 0)
q(){ "$CH_CMD" "${OPTS[@]}" --query "$1" 2>&1 | grep -vi "unknown setting" || true; }
n(){ "$CH_CMD" "${OPTS[@]}" --format TSV --query "$1" 2>/dev/null; }
q "CREATE DATABASE IF NOT EXISTS bench"

# ---- pool + queries + ground truth for a non-photos model -------------------------------------------
# args: label  source-table  key-expr  embedding-expr  sample-stride
setup_pool(){ L=$1; SRC=$2; KEY=$3; EMB=$4; STRIDE=$5
  q "DROP TABLE IF EXISTS bench.samp_$L"
  q "CREATE TABLE bench.samp_$L (md5 String, embedding Array(BFloat16)) ENGINE = MergeTree ORDER BY md5"
  q "INSERT INTO bench.samp_$L
       SELECT $KEY AS md5, $EMB AS embedding FROM $SRC WHERE cityHash64($KEY) % $STRIDE = 0"
  echo "  pool $L: samp=$(n "SELECT count() FROM bench.samp_$L")"
}

# ---- query points + ground truth, in the SEARCH's own precision ------------------------------------
# Reads bench.cs_<label>_rabitq (the codec pool). With vector_search_use_quantized_codes left at 0 this
# is a plain exact scan over the full-precision vectors the codec stores alongside its codes, so the
# distances are bit-identical to what the benchmark computes. The turboquant pool holds the same
# `embedding` values (same source, codec preserves full precision), so one ground truth serves both.
# args: label
setup_gt(){ L=$1
  P="bench.cs_${L}_rabitq"
  q "DROP TABLE IF EXISTS bench.qc_$L"; q "DROP TABLE IF EXISTS bench.gtc_$L"

  q "CREATE TABLE bench.qc_$L (q_idx UInt32, q_md5 String, ref Array(BFloat16))
     ENGINE = MergeTree ORDER BY q_idx"
  q "INSERT INTO bench.qc_$L
       SELECT rowNumberInAllBlocks(), md5, embedding
       FROM (SELECT md5, embedding FROM $P ORDER BY cityHash64(md5) LIMIT $NQ)"

  q "CREATE TABLE bench.gtc_$L (q_idx UInt32, gt10 Array(String), gt100 Array(String))
     ENGINE = MergeTree ORDER BY q_idx"
  q "INSERT INTO bench.gtc_$L
       SELECT q_idx,
              arraySlice(arrayMap(t->t.2, arraySort(t->t.1, groupArray((dist,md5)))),1,10),
              arrayMap(t->t.2, arraySort(t->t.1, groupArray((dist,md5))))
       FROM (SELECT q.q_idx q_idx, s.md5 md5, cosineDistance(s.embedding, q.ref) dist
             FROM $P s CROSS JOIN bench.qc_$L q WHERE s.md5 != q.q_md5
             ORDER BY q_idx, dist ASC LIMIT 100 BY q_idx) GROUP BY q_idx"
  echo "  gt $L: qc=$(n "SELECT count() FROM bench.qc_$L") gtc=$(n "SELECT count() FROM bench.gtc_$L")"
}

# ---- codec copies of a pool -------------------------------------------------------------------------
# DEDUPED BY EMBEDDING first. The web pools carry a lot of byte-identical vectors (near-duplicate pages:
# 11.5% of web_txt_bge_large, 6.1% of web_img_siglip2), which produce exact distance ties. Neither the
# ground-truth query nor the benchmark's search can carry a tiebreak -- the codec optimisation only fires
# on the bare `ORDER BY <distance> LIMIT k` shape -- so which of a set of tied rows lands at position 100
# is arbitrary and varies between runs. That alone made the codec-disabled control score 0.916 recall@10
# on web_txt_bge_large. Deduping removes the ambiguity; duplicate pages are a corpus artifact, not
# something a recall benchmark should be measuring. `ORDER BY md5 LIMIT 1 BY embedding` keeps the
# representative deterministic, so both codec pools hold the same rows and can share one ground truth.
# args: label  source-samp-table  dim
setup_codec(){ L=$1; SAMP=$2; D=$3
  q "DROP TABLE IF EXISTS bench.sampd_$L"
  q "CREATE TABLE bench.sampd_$L (md5 String, embedding Array(BFloat16)) ENGINE = MergeTree ORDER BY md5"
  q "INSERT INTO bench.sampd_$L SELECT md5, embedding FROM (SELECT md5, embedding FROM $SAMP ORDER BY md5) LIMIT 1 BY embedding"
  echo "  dedup $L: $(n "SELECT count() FROM bench.sampd_$L") of $(n "SELECT count() FROM $SAMP") rows kept"
  for C in rabitq turboquant; do
    q "DROP TABLE IF EXISTS bench.cs_${L}_$C"
    q "CREATE TABLE bench.cs_${L}_$C (md5 String, embedding Array(BFloat16) CODEC(Quantized('$C', $D, 0)))
       ENGINE = MergeTree ORDER BY md5
       SETTINGS index_granularity = 8192, min_bytes_for_wide_part = 0, min_rows_for_wide_part = 0"
    q "INSERT INTO bench.cs_${L}_$C SELECT md5, embedding FROM bench.sampd_$L"
    echo "  codec cs_${L}_$C: $(n "SELECT count() FROM bench.cs_${L}_$C") rows"
  done
}

echo "=== photos: reusing existing bench.samp_emb_* pools, adding codec copies"
setup_codec siglip2 bench.samp_emb_siglip2 1152
setup_codec clip    bench.samp_emb_clip     768
setup_codec nomic   bench.samp_emb_nomic    768

echo "=== web: new pools (17.83M / 18.99M rows -> stride for ~100k)"
setup_pool web_img_siglip2   default.web_emb_img_siglip2   url "CAST(embedding,'Array(BFloat16)')" 178
setup_pool web_txt_bge_large default.web_emb_txt_bge_large url "CAST(embedding,'Array(BFloat16)')" 190
setup_codec web_img_siglip2   bench.samp_web_img_siglip2   1152
setup_codec web_txt_bge_large bench.samp_web_txt_bge_large 1024

echo "=== hn: new pools (37.80M rows -> stride for ~100k); embedding is QBit -> CAST to Array"
setup_pool hn_nomic     default.hackernews_embeddings_nomic     "toString(id)" "CAST(embedding,'Array(BFloat16)')" 378
setup_pool hn_bge_large default.hackernews_embeddings_bge_large "toString(id)" "CAST(embedding,'Array(BFloat16)')" 378
setup_codec hn_nomic     bench.samp_hn_nomic      768
setup_codec hn_bge_large bench.samp_hn_bge_large 1024

echo "=== query points + ground truth ($NQ queries per pool, search-precision)"
for L in siglip2 clip nomic web_img_siglip2 web_txt_bge_large hn_nomic hn_bge_large; do setup_gt "$L"; done

echo "setup done — now run: python3 scripts/recall_codecs_bench.py"
