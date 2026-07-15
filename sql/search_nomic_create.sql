CREATE TABLE IF NOT EXISTS mmcommons.search_nomic
(
    `photo_id` UInt64,
    `user_nsid` String,
    `user_nickname` LowCardinality(String),
    `date_taken` Nullable(DateTime64(0)),
    `date_uploaded` DateTime,
    `capture_device` LowCardinality(String),
    `title` String,
    `description` String,
    `user_tags` Array(String),
    `machine_tags` Array(String),
    `lon` Nullable(Float64),
    `lat` Nullable(Float64),
    `geo_accuracy` UInt8,
    `page_url` String,
    `download_url` String,
    `license_name` LowCardinality(String),
    `license_url` LowCardinality(String),
    `server_id` UInt32,
    `farm_id` UInt16,
    `secret` String,
    `secret_original` String,
    `ext` LowCardinality(String),
    `media` Enum8('photo' = 0, 'video' = 1),
    `md5` String MATERIALIZED lowerUTF8(hex(MD5(download_url))),
    `new_url` String DEFAULT replaceRegexpOne(download_url, '^.*/([^/]+)/([^/]+)$', 'https://live.staticflickr.com/\\1/\\2'),
    `s3_url` String DEFAULT concat('https://multimedia-commons.s3.us-west-2.amazonaws.com/data/images/', substring(arrayStringConcat(arrayMap(i -> if(substring(lower(hex(MD5(download_url))) AS hx, (i * 2) - 1, 1) = '0', substring(hx, i * 2, 1), substring(hx, (i * 2) - 1, 2)), range(1, 17))) AS h, 1, 3), '/', substring(h, 4, 3), '/', h, '.jpg'),
    `embedding` QBit(BFloat16, 768, 16)
)
ENGINE = SharedMergeTree
ORDER BY photo_id
SETTINGS index_granularity = 8192;
