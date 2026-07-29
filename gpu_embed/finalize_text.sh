#!/bin/bash
# Build hackernews_embeddings_<model> with the same schema as hackernews_embeddings_qwen3_8b
# (metadata columns + QBit embedding + strided/int/rotated derived columns), populated by
# joining the per-model staging vectors to the Qwen3 table's metadata on `id`.
# Usage: CH_HOST=.. CH_PW=.. ./finalize_text.sh <model> [--keep-stage]
set -e
: "${CH_HOST:?}"; : "${CH_PW:?}"
MODEL=${1:?usage: ./finalize_text.sh <model>}
KEEP=${2:-}
PY="$([ -x "$PWD/venv/bin/python" ] && echo "$PWD/venv/bin/python" || echo python3)"
DIM=$("$PY" -c "from text_models import MODELS; print(MODELS['$MODEL']['dim'])")
SRC=${SRC_TABLE:-hackernews_embeddings_qwen3_8b}
DST=hackernews_embeddings_${MODEL}
PORT=${CH_PORT:-8443}
ch() { curl -sS "https://${CH_HOST}:${PORT}/" --user "${CH_USER:-default}:${CH_PW}" --data-binary "$1"; }

echo "=== finalize $DST  (dim=$DIM, src=$SRC) ==="
STAGE_N=$(ch "SELECT count() FROM stage_hn_${MODEL}" 2>/dev/null || echo 0)
echo "stage rows: $STAGE_N"
if ! [[ "$STAGE_N" =~ ^[0-9]+$ ]] || [ "$STAGE_N" = "0" ]; then
  echo "!! staging table stage_hn_${MODEL} missing or empty — skipping $MODEL (worker likely failed to load)."
  exit 1
fi

# Only the base embedding column. Derived/quantized/rotated columns are intentionally omitted:
# Int8 quantization depends on the normalization scheme and rotation isn't always needed
# (e.g. MRL embeddings) — those can be materialized later per-model.
ch "CREATE TABLE IF NOT EXISTS ${DST}
(
    \`update_time\` DateTime,
    \`id\` UInt32,
    \`deleted\` UInt8,
    \`type\` Enum8('story' = 1, 'comment' = 2, 'poll' = 3, 'pollopt' = 4, 'job' = 5),
    \`by\` LowCardinality(String),
    \`time\` DateTime,
    \`text\` String,
    \`dead\` UInt8,
    \`parent\` UInt32,
    \`poll\` UInt32,
    \`kids\` Array(UInt32),
    \`url\` String,
    \`score\` Int32,
    \`title\` String,
    \`parts\` Array(UInt32),
    \`descendants\` Int32,
    \`embedding\` QBit(BFloat16, ${DIM}) CODEC(ZSTD(3))
)
ENGINE = SharedMergeTree('/clickhouse/tables/{uuid}/{shard}', '{replica}')
ORDER BY id
SETTINGS index_granularity = 8192" | head -5

DST_N=$(ch "SELECT count() FROM ${DST}")
if [ "$DST_N" != "0" ]; then
  echo "$DST already has $DST_N rows; skipping populate (drop it to rebuild)."
else
  echo "populating $DST from ${SRC} JOIN stage_hn_${MODEL} ..."
  ch "INSERT INTO ${DST}
      (update_time,id,deleted,type,by,time,text,dead,parent,poll,kids,url,score,title,parts,descendants,embedding)
      SELECT h.update_time,h.id,h.deleted,h.type,h.by,h.time,h.text,h.dead,h.parent,h.poll,h.kids,h.url,h.score,h.title,h.parts,h.descendants,
             s.vector
      FROM ${SRC} AS h INNER JOIN stage_hn_${MODEL} AS s USING (id)
      SETTINGS join_algorithm='full_sorting_merge', max_execution_time=0" | head -5
  DST_N=$(ch "SELECT count() FROM ${DST}")
fi
echo "final rows in $DST: $DST_N   (dim check: $(ch "SELECT length(CAST(embedding AS Array(BFloat16))) FROM ${DST} LIMIT 1"))"

if [ "$KEEP" != "--keep-stage" ] && [ "$DST_N" = "$STAGE_N" ] && [ "$DST_N" != "0" ]; then
  echo "dropping staging table stage_hn_${MODEL}"
  ch "DROP TABLE stage_hn_${MODEL}" | head -2
fi
echo "=== $DST done ==="
