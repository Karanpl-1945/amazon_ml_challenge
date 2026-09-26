"""Clean every source file (train + test) with the mined dictionaries and write parquet.

Output: config.CLEAN_DIR/{train,test}_{s1,s2,s3}.parquet with the raw columns plus the structured
fields produced by normalize.Normalizer.clean_record. Train S1 also gets `is_val` (the 20% fold).

Memory-safe: rows are sliced from the Arrow table one wave (= `workers` chunks) at a time, workers
return compact column lists, and each file is written to <name>.tmp and renamed only when complete.
Existing complete files are skipped unless --force.

Run (after mine_dictionaries.py):  python src/clean_data.py [--workers 12] [--only test] [--force]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

import config
from io_utils import read_tsv_table
from normalize import Normalizer

CHUNK = 50_000
_NORM = None


def _init_worker():
    global _NORM
    _NORM = Normalizer(config.DICT_DIR)


def _clean_chunk(cols):
    """cols: (names, addresses, countries) lists -> dict of output columns (lists)."""
    names, addrs, ctrys = cols
    recs = [_NORM.clean_record(n, a, c) for n, a, c in zip(names, addrs, ctrys)]
    return {k: [r[k] for r in recs] for k in recs[0]}


def _is_complete(path, n_rows):
    try:
        return pq.ParquetFile(path).metadata.num_rows == n_rows
    except Exception:
        return False


def clean_file(src_path, dst_path, pool, workers, add_val_flag=False, force=False):
    t0 = time.time()
    tbl = read_tsv_table(src_path)
    if not force and dst_path.exists() and _is_complete(dst_path, tbl.num_rows):
        print(f"  {dst_path.name}: already complete, skipping")
        return
    tmp = dst_path.with_suffix(".tmp")
    starts = list(range(0, tbl.num_rows, CHUNK))
    writer = None
    for w in range(0, len(starts), workers):
        wave = [tbl.slice(s, CHUNK) for s in starts[w:w + workers]]
        payload = [(p["business_name"].to_pylist(), p["business_address"].to_pylist(), p["country"].to_pylist())
                   for p in wave]
        for part, res in zip(wave, pool.map(_clean_chunk, payload)):
            out = part
            for k, v in res.items():
                out = out.append_column(k, pa.array(v))
            if add_val_flag:
                out = out.append_column("is_val", pa.array([config.is_val_entity(e) for e in part["entity_id"].to_pylist()]))
            if writer is None:
                writer = pq.ParquetWriter(tmp, out.schema, compression="zstd")
            writer.write_table(out)
        del wave, payload
    writer.close()
    os.replace(tmp, dst_path)
    print(f"  {src_path.name}: {tbl.num_rows:,} rows -> {dst_path.name}  ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", choices=["train", "test"], default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    jobs = []
    if args.only in (None, "train"):
        jobs += [(config.TRAIN_FILES[s], config.CLEAN_DIR / f"train_{s}.parquet", s == "s1") for s in ("s1", "s2", "s3")]
    if args.only in (None, "test"):
        jobs += [(config.TEST_FILES[s], config.CLEAN_DIR / f"test_{s}.parquet", False) for s in ("s1", "s2", "s3")]
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for src, dst, flag in jobs:
            clean_file(src, dst, pool, args.workers, add_val_flag=flag, force=args.force)
    print(f"all done in {time.time()-t0:.0f}s -> {config.CLEAN_DIR}")


if __name__ == "__main__":
    sys.exit(main())
