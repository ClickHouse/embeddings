cd /home/ubuntu/embeddings
./clickhouse-new local --path . --query "
CREATE OR REPLACE TABLE hn_meta ENGINE=MergeTree ORDER BY id AS
SELECT id,
  max(ut) AS update_time,
  argMax(deleted,ut) AS deleted, argMax(type,ut) AS type, argMax(by,ut) AS by,
  argMax(time,ut) AS time, argMax(text,ut) AS text, argMax(dead,ut) AS dead,
  argMax(parent,ut) AS parent, argMax(poll,ut) AS poll, argMax(kids,ut) AS kids,
  argMax(url,ut) AS url, argMax(score,ut) AS score, argMax(title,ut) AS title,
  argMax(parts,ut) AS parts, argMax(descendants,ut) AS descendants
FROM (SELECT *, update_time AS ut FROM hackernews_history WHERE type='comment' AND deleted=0 AND dead=0 AND notEmpty(text))
GROUP BY id
SETTINGS max_threads=48, max_insert_threads=8"
echo "META_RC=$?"
