#!/bin/bash
# Run the light models SEQUENTIALLY, cheapest first, each as a full parallel pass over all GPUs.
# Rationale (all measured on 4x L4):
#   * cheapest-first (nomic ViT-B < clip ViT-L < siglip2 so400m@512) => the first complete, usable
#     embedding table lands soonest; total wall-time is ~unchanged vs interleaving (GPU work is the same).
#   * 1 process per GPU is the sweet spot. 2 procs/GPU collapses throughput ~5x: two CUDA contexts on
#     one L4 (no MPS) serialize kernels with costly context switches (measured 351 -> 65 img/s).
#   * each pass re-reads S3, but that's ~free here (in-region bucket, CPU mostly idle) and the run is
#     GPU/pipeline-bound, not decode-bound at 1 proc/GPU.
# Resumable: per-model .done markers mean a killed/restarted pass skips finished prefixes.
# Nemotron is intentionally excluded — it's ~5x heavier and belongs on faster GPUs (g6e/L40S); run it
# separately with:  ./run.sh <ngpu> nemotron
set -e
NGPU=${1:-$(nvidia-smi -L | wc -l)}
run_model(){ echo "==== $(date '+%F %T')  START $1  (batch $2, $NGPU GPUs x1) ===="; \
             PROCS_PER_GPU=1 ./run.sh "$NGPU" "$1" "$2"; \
             echo "==== $(date '+%F %T')  DONE  $1 ===="; }
run_model nomic   512   # cheapest: ViT-B/16 @224
run_model clip    512   # ViT-L/14 @224
run_model siglip2 256   # priciest light model: so400m @512 (1024 patches) -> smaller batch
echo "ALL LIGHT MODELS COMPLETE $(date '+%F %T')"
