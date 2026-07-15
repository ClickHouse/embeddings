#!/bin/bash
cd /home/ubuntu/embeddings
FB=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks
AZ=https://comp21storage.z5.web.core.windows.net
run(){ echo "=== $(date +%T) START $1 ($6 rows) ==="; bash annrun.sh "$@"; echo "=== $(date +%T) END $1 ==="; }
#   table url typ dim rowbytes N chunkrows W
run yfcc10m_base                "$FB/yfcc100M/base.10M.u8bin"                                    u8  192  192  10000000   5000000 24
run openai2m_base               "$AZ/arxiv-openaiv2-2M/openai_base.bin"                          f32 1536 6144 2321096    250000  24
run caselaw7m_base              "$AZ/caselaw/single_vector/caselaw_base_embeddings.bin"          f32 1536 6144 7414023    250000  24
run msturing10m_clustered_base  "$AZ/comp23/clustered_data/msturing-10M-clustered/msturing-10M-clustered.fbin" f32 100 400 10000000 5000000 24
run msturing30m_clustered_base  "$AZ/comp23/clustered_data/msturing-30M-clustered/30M-clustered64.fbin"        f32 100 400 29998994 5000000 24
run wikipedia35m_base           "$AZ/wiki-cohere-35M/wikipedia_base.bin"                         f32 768  3072 35000000   500000  24
run ssnpp1b_base                "$FB/FB_ssnpp_database.u8bin"                                    u8  256  256  1000000000 5000000 24
run msmarco100m_base            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/vectors/SimANS/passage_vectors/vectors.bin" f32 768 3072 101070374 500000 24
run yfccimages100m_base         "$AZ/yfcc100m_images/yfcc100m_vecs.fbin"                         f32 1280 5120 98735605   400000  24
echo "ALL NEW BASE DONE"
