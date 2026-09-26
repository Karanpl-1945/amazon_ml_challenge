"""Score the test candidates and write the two submission files.

Reads  : work/feat/test_*.parquet, work/model/lgb.txt, work/model/decision.json, test S1 ids
Writes : output/matching_results.tsv   (scored on the leaderboard)
         output/candidate_pairs.tsv    (exactly the pairs the model ran inference on)
Every test Source-1 entity gets one row in both files (empty list when nothing matched / no candidates).
ID lists are ordered by model confidence. One feature file (= one country) is processed at a time and
lists are built with Arrow, so memory stays low; candidate ids never repeat across countries, so the
best-S1 assignment per candidate can be applied per file.

Run: python src/predict.py [--threads 8]
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config
from features import FEAT_DIR
from io_utils import read_tsv_table
from train_matcher import MODEL_DIR, assign_best, to_matrix


def joined_lists(s1, ids):
    """(s1 array sorted by s1, id array) -> {s1: 'id1,id2,...'} built with Arrow list ops."""
    if len(s1) == 0:
        return {}
    codes = pc.dictionary_encode(s1).indices.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    offsets = pa.array(np.r_[starts, len(codes)].astype(np.int32))
    lists = pa.ListArray.from_arrays(offsets, ids)
    joined = pc.binary_join(lists, ",")
    return dict(zip(s1.take(pa.array(starts)).to_pylist(), joined.to_pylist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 4))
    args = ap.parse_args()
    t0 = time.time()
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgb.txt"))
    decision = json.load(open(MODEL_DIR / "decision.json"))
    names = decision["features"]
    files = sorted(FEAT_DIR.glob("test_*.parquet"))
    if not files:
        sys.exit(f"no feature files in {FEAT_DIR} - run features.py --split test first")

    cand_str, match_str, n_pairs, n_match = {}, {}, 0, 0
    for f in files:
        s1, cd, prob = [], [], []
        for b in pq.ParquetFile(f).iter_batches(batch_size=1_000_000, columns=names + ["s1_id", "cand_id"]):
            t = pa.Table.from_batches([b])
            prob.append(booster.predict(to_matrix(t, names), num_threads=args.threads))
            s1.append(t["s1_id"].combine_chunks())
            cd.append(t["cand_id"].combine_chunks())
        t = pa.table({"s1_id": pa.concat_arrays(s1), "cand_id": pa.concat_arrays(cd), "prob": np.concatenate(prob)})
        del s1, cd, prob
        keep = assign_best(t["cand_id"].combine_chunks(), t["prob"].to_numpy()) if decision["assign_best"] \
            else np.ones(t.num_rows, bool)
        t = t.append_column("match", pa.array(keep & (t["prob"].to_numpy() >= decision["threshold"])))
        t = t.take(pc.sort_indices(t, [("s1_id", "ascending"), ("prob", "descending")]))
        cand_str.update(joined_lists(t["s1_id"].combine_chunks(), t["cand_id"].combine_chunks()))
        m = t.filter(t["match"])
        match_str.update(joined_lists(m["s1_id"].combine_chunks(), m["cand_id"].combine_chunks()))
        n_pairs += t.num_rows
        n_match += m.num_rows
        print(f"  scored {f.name}: {t.num_rows:,} pairs, {m.num_rows:,} matches  ({time.time()-t0:.0f}s)", flush=True)
        del t, m

    test_s1 = read_tsv_table(config.TEST_FILES["s1"]).column("entity_id").to_pylist()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, header, lists in (("matching_results.tsv", "matched_entity_ids", match_str),
                                 ("candidate_pairs.tsv", "candidate_entity_ids", cand_str)):
        with open(config.OUTPUT_DIR / fname, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"source1_entity_id\t{header}\n")
            for s in test_s1:
                fh.write(f"{s}\t{lists.get(s, '')}\n")
    n_with = sum(1 for s in test_s1 if s in match_str)
    print(f"{len(test_s1):,} test S1 entities | {n_pairs:,} candidate pairs | {n_match:,} matches | "
          f"{n_with:,} entities with >=1 match ({n_with/len(test_s1):.1%})")
    print(f"wrote {config.OUTPUT_DIR / 'matching_results.tsv'} and candidate_pairs.tsv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
