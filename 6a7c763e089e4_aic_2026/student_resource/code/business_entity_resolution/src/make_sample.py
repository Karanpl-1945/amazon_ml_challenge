"""Build a small, same-format copy of the dataset for a quick end-to-end smoke test of the pipeline.

A deterministic fraction of Source-1 entities is kept per split; for train, all of their true matches
are kept plus the same fraction of every other S2/S3 record (distractors); for test, the same fraction
of S2/S3 is kept at random. The ground truth is filtered to the kept S1 entities.

Run:  python src/make_sample.py --out <dir> [--frac 0.005]
Then: set ER_DATA_DIR=<dir>, ER_WORK_DIR=<dir>/work, ER_OUTPUT_DIR=<dir>/output and run run_pipeline.py
"""
import argparse
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import config
from io_utils import read_tsv_table


def _pick(ids, frac, salt):
    thr = int(frac * 1_000_000)
    return np.array([zlib.crc32((salt + e).encode()) % 1_000_000 < thr for e in ids.to_pylist()], bool)


def _write(tbl, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [tbl[c].to_pylist() for c in tbl.column_names]     # same format as the originals: raw, unquoted
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(tbl.column_names) + "\n")
        for row in zip(*cols):
            fh.write("\t".join(row) + "\n")
    print(f"  {path.name}: {tbl.num_rows:,} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.005)
    args = ap.parse_args()
    out = Path(args.out)

    s1 = read_tsv_table(config.TRAIN_FILES["s1"])
    s1 = s1.filter(pa.array(_pick(s1["entity_id"], args.frac, "s1:")))
    _write(s1, out / "train" / "train_source1.tsv")
    gt = read_tsv_table(config.TRAIN_GT)
    gt = gt.filter(pc.is_in(gt["source1_entity_id"], value_set=s1["entity_id"]))
    _write(gt, out / "train" / "train_ground_truth.tsv")
    linked = pc.list_flatten(pc.split_pattern(gt["matched_entity_ids"], ","))
    for s in ("s2", "s3"):
        t = read_tsv_table(config.TRAIN_FILES[s])
        m = pc.is_in(t["entity_id"], value_set=linked).to_numpy(zero_copy_only=False) | _pick(t["entity_id"], args.frac, s)
        _write(t.filter(pa.array(m)), out / "train" / f"train_source{s[1]}.tsv")
    for s in ("s1", "s2", "s3"):
        t = read_tsv_table(config.TEST_FILES[s])
        _write(t.filter(pa.array(_pick(t["entity_id"], args.frac, "t" + s))), out / "test" / f"test_source{s[1]}.tsv")


if __name__ == "__main__":
    main()
