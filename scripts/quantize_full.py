"""Parallel streaming quantizer: clickhouse-local (Arrow on stdin) -> process pool -> Int8 shards.
Each worker quantizes one Arrow batch and writes shard_<i>.parquet (FixedSizeList<int8>[4096])."""
import sys, os, numpy as np, pyarrow as pa, pyarrow.parquet as pq
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import quant

_SIGNS = quant.rotation_signs()
OUT = "qcodes"

def work(args):
    i, ids_bytes, vbytes, n = args
    ids = np.frombuffer(ids_bytes, dtype=np.uint32)
    X = np.frombuffer(vbytes, dtype=np.float32).reshape(n, 4096)
    idx = quant.quantize(X, _SIGNS)                                  # uint8 0..255
    code = (idx.astype(np.int16) - 128).astype(np.int8).reshape(-1)  # -128..127
    tbl = pa.table({"id": pa.array(ids),
                    "q": pa.FixedSizeListArray.from_arrays(pa.array(code), 4096)})
    pq.write_table(tbl, f"{OUT}/shard_{i:05d}.parquet", compression="zstd")
    return n

def chunks(target=32768):
    """Read Arrow batches, coalesce to ~target rows/chunk so shards aren't tiny."""
    reader = pa.ipc.open_stream(sys.stdin.buffer)
    idb, vb, n, i = [], [], 0, 0
    def emit():
        nonlocal idb, vb, n, i
        ids = np.concatenate(idb); v = np.concatenate(vb)
        out = (i, ids.tobytes(), v.tobytes(), len(ids)); i += 1
        idb, vb, n = [], [], 0
        return out
    for b in reader:
        idb.append(b.column("id").to_numpy().astype(np.uint32))
        vb.append(b.column("v").values.to_numpy(zero_copy_only=True).astype(np.float32, copy=False))
        n += len(idb[-1])
        if n >= target:
            yield emit()
    if n:
        yield emit()

def main():
    global OUT
    OUT = sys.argv[1] if len(sys.argv) > 1 else "qcodes"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    os.makedirs(OUT, exist_ok=True)
    total, last = 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:   # fork: inherits OUT/_SIGNS, no re-import
        inflight = deque()
        for item in chunks(32768):
            inflight.append(ex.submit(work, item))
            if len(inflight) >= workers * 2:
                total += inflight.popleft().result()
                if total - last >= 2_000_000:
                    last = total; print(f"  {total:,} rows", flush=True)
        for f in inflight:
            total += f.result()
    print(f"DONE total={total:,}", flush=True)

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
