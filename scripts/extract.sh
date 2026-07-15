cd /home/ubuntu/embeddings
./clickhouse-new local --path . --query "
INSERT INTO FUNCTION file('hn_in/shard_{_partition_id}.parquet', Parquet)
PARTITION BY (cityHash64(id) % 512)
SELECT id, trimBoth(decodeHTMLComponent(extractTextFromHTML(concat(ttl,' ',txt)))) AS doc
FROM (
  SELECT id, argMax(text,update_time) AS txt, argMax(title,update_time) AS ttl
  FROM hackernews_history
  WHERE type='comment' AND deleted=0 AND dead=0 AND notEmpty(text)
  GROUP BY id
) WHERE notEmpty(doc)
SETTINGS max_threads=48, max_insert_threads=8, engine_file_truncate_on_insert=1"
echo "EXTRACT_RC=$?"
