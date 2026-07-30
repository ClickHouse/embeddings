#!/usr/bin/env bash
# Populate default.web_nsfw (url, nsfw Float32) = P(adult website) for all ~17.83M WonderfulWeb sites, by
# applying the logistic-regression NSFW classifier (trained by nsfw_train.py on SigLIP2 image embeddings;
# weights in nsfw_weights.txt / nsfw_bias.txt) as dotProduct + sigmoid over web_emb_img_siglip2. ~6s.
# Held-out AUC 0.989. Score generalises past the domain-labelled training seeds (catches adult sites by look).
set -uo pipefail
H=https://hvdvsqo23t.us-east-2.aws.clickhouse-staging.com:8443
DIR="$(cd "$(dirname "$0")" && pwd)"; W=$(cat "$DIR/nsfw_weights.txt"); B=$(cat "$DIR/nsfw_bias.txt")
curl -s "$H/?user=default&password=$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --data-binary \
  "CREATE TABLE IF NOT EXISTS default.web_nsfw (url String, nsfw Float32) ENGINE = MergeTree ORDER BY url"
curl -s "$H/?user=default&password=$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD&max_execution_time=0" --data-binary \
  "INSERT INTO default.web_nsfw SELECT url, toFloat32(1/(1+exp(-(dotProduct(CAST(embedding,'Array(Float32)'),[$W]) + ($B))))) FROM default.web_emb_img_siglip2"
echo "done"
