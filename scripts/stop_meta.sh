#!/bin/bash
# Stop ONLY metadata pushes (sqlite3 + the yfcc_metadata insert clients); leave the feature driver alone.
pkill -9 sqlite3 2>/dev/null
pkill -9 -f 'mmcommons.yfcc_metadata' 2>/dev/null
pkill -9 -f 'push_yfcc_meta' 2>/dev/null
sleep 2
echo "stopped. sqlite3=$(pgrep -fc sqlite3) meta_insert=$(pgrep -af 'mmcommons.yfcc_metadata'|grep -vc grep)"
echo "feature driver alive=$(pgrep -fc ingest_features.sh) feat_insert=$(pgrep -af 'mmcommons.feat_'|grep -vc grep)"
