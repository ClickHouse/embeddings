#!/bin/bash
cd /home/ubuntu/embeddings
DB=mmcommons/metadata/yfcc100m_dataset.sql
COLS="photoid,uid,unickname,datetaken,dateuploaded,capturedevice,title,description,usertags,machinetags,longitude,latitude,accuracy,pageurl,downloadurl,licensename,licenseurl,serverid,farmid,secret,secretoriginal,ext,marker"
sqlite3 -separator $'\t' "$DB" "SELECT $COLS FROM yfcc100m_dataset" > mmcommons/meta.tsv
echo "DUMP_RC=$? rows=$(wc -l < mmcommons/meta.tsv) $(date -u +%H:%M:%S)" > mmcommons/meta_dump.done
