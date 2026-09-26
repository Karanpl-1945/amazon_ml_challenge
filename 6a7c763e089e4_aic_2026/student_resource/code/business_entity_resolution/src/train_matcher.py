"""Train the pair matcher (LightGBM, MIT licence) and tune the decision rule for macro F0.5.

Training rows  : featurised candidate pairs of the train-fold sample (is_val == False).
Early stopping : candidates of 25% of the validation-fold entities.
Decision rule  : tuned on ALL validation-fold entities, scored exactly like the leaderboard
                 (F0.5 per Source-1 entity - singletons and entities without candidates included - averaged).
  1. (optional) each S2/S3 record is kept only for the S1 entity that gives it the highest probability
     (the ground truth links every S2/S3 record to at most one S1 entity)
  2. a pair is a match if its probability >= threshold
Validation pairs are streamed in batches, so memory stays ~ the training matrix.

Output: work/model/lgb.txt, work/model/decision.json
Run: python src/train_matcher.py [--threads 8] [--max-train-rows 8000000]
"""
import argparse
import json
import os
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config
from blocking import gt_pair_table
from features import FEAT_DIR

MODEL_DIR = config.WORK_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
NON_FEATURES = {"s1_id", "cand_id", "label", "is_val"}

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=config.SEED)


def feature_names(schema):
    return [n for n in schema.names if n not in NON_FEATURES]


def to_matrix(tbl, names):
    X = np.empty((tbl.num_rows, len(names)), np.float32)
    for j, n in enumerate(names):
        X[:, j] = tbl[n].to_numpy(zero_copy_only=False)
    return X


def assign_best(cand_ids, prob):
    """Mask keeping, for every candidate id, only its highest-probability row."""
    cd = pc.dictionary_encode(cand_ids).indices.to_numpy()
    order = np.lexsort((-prob, cd))
    first = np.r_[True, np.diff(cd[order]) != 0]
    keep = np.zeros(len(prob), bool)
    keep[order[first]] = True
    return keep


def macro_f05(n_true, q_pred, correct_pred):
    """Leaderboard metric, vectorised. n_true[i] = #true matches of evaluated S1 entity i (all of them,
    singletons included); q_pred / correct_pred: per predicted pair, its entity index and correctness."""
    n = len(n_true)
    p = np.bincount(q_pred, minlength=n)
    ok = np.bincount(q_pred[correct_pred], minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr, rc = ok / p, ok / n_true
        f = np.where(ok > 0, np.nan_to_num(1.25 * pr * rc / (0.25 * pr + rc)), 0.0)
    f = np.where((n_true == 0) & (p == 0), 1.0, f)
    return float(f.mean())


def _stop_subset(s1_id):
    return zlib.crc32(("stop:" + s1_id).encode("utf-8")) % 4 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--max-train-rows", type=int, default=8_000_000,
                    help="cap on training pairs (random rows) to bound RAM")
    ap.add_argument("--rounds", type=int, default=3000)
    args = ap.parse_args()
    PARAMS["num_threads"] = args.threads
    t0 = time.time()
    files = sorted(FEAT_DIR.glob("train_*.parquet"))
    if not files:
        sys.exit(f"no feature files in {FEAT_DIR} - run features.py --split train first")
    names = feature_names(pq.read_schema(files[0]))

    # ---- training matrix (train-fold sample)
    Xs, ys = [], []
    for f in files:
        t = pq.read_table(f, columns=names + ["label"], filters=[("is_val", "=", False)])
        Xs.append(to_matrix(t, names))
        ys.append(t["label"].to_numpy())
    X, y = np.concatenate(Xs), np.concatenate(ys)
    del Xs, ys
    if len(y) > args.max_train_rows:
        idx = np.random.default_rng(config.SEED).choice(len(y), args.max_train_rows, replace=False)
        X, y = X[idx], y[idx]

    # ---- early-stopping matrix (25% of validation entities)
    Xv, yv = [], []
    for f in files:
        t = pq.read_table(f, columns=names + ["label", "s1_id"], filters=[("is_val", "=", True)])
        m = np.array([_stop_subset(s) for s in t["s1_id"].to_pylist()], bool)
        t = t.filter(pa.array(m))
        Xv.append(to_matrix(t, names))
        yv.append(t["label"].to_numpy())
    Xv, yv = np.concatenate(Xv), np.concatenate(yv)
    print(f"train {len(y):,} pairs (pos {y.mean():.3f}), early-stop {len(yv):,} pairs, "
          f"{len(names)} features  ({time.time()-t0:.0f}s)", flush=True)

    dtr = lgb.Dataset(X, y, feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    booster = lgb.train(PARAMS, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    del dtr, dva, X, y, Xv, yv
    booster.save_model(str(MODEL_DIR / "lgb.txt"))
    imp = sorted(zip(names, booster.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", ", ".join(f"{n}={g:.0f}" for n, g in imp[:15]), flush=True)

    # ---- decision rule on the whole validation fold, streamed
    s1_all = pq.read_table(config.CLEAN_DIR / "train_s1.parquet", columns=["entity_id", "is_val"])
    val_ids = s1_all.filter(s1_all["is_val"])["entity_id"].combine_chunks()
    gt = gt_pair_table()
    gt_q = pc.index_in(gt["s1_id"], value_set=val_ids).drop_null().to_numpy()
    n_true = np.bincount(gt_q, minlength=len(val_ids))

    q, cd, prob, lab = [], [], [], []
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=1_000_000, columns=names + ["s1_id", "cand_id", "label", "is_val"]):
            t = pa.Table.from_batches([b])
            t = t.filter(t["is_val"])
            if t.num_rows == 0:
                continue
            prob.append(booster.predict(to_matrix(t, names), num_threads=args.threads))
            q.append(pc.index_in(t["s1_id"], value_set=val_ids).to_numpy())
            cd.append(t["cand_id"].combine_chunks())
            lab.append(t["label"].to_numpy().astype(bool))
    q, prob, lab = np.concatenate(q), np.concatenate(prob), np.concatenate(lab)
    cd = pa.concat_arrays(cd)

    best = {"f05": -1.0}
    grid = []
    for use_assign in (False, True):
        keep = assign_best(cd, prob) if use_assign else np.ones(len(prob), bool)
        for th in np.round(np.arange(0.30, 0.96, 0.05), 2):
            m = keep & (prob >= th)
            f = macro_f05(n_true, q[m], lab[m])
            grid.append({"assign_best": use_assign, "threshold": float(th), "f05": round(f, 5)})
            print(f"  assign_best={use_assign!s:5}  threshold={th:.2f}  val macro F0.5={f:.4f}", flush=True)
            if f > best["f05"]:
                best = {"f05": f, "threshold": float(th), "assign_best": use_assign}
    best["blocking_ceiling_f05"] = macro_f05(n_true, q[lab], lab[lab])  # perfect matcher on these candidates
    best["val_entities"] = len(val_ids)
    best["best_iteration"] = booster.best_iteration
    best["features"] = names
    best["grid"] = grid
    json.dump(best, open(MODEL_DIR / "decision.json", "w"), indent=1)
    print(f"BEST val macro F0.5 = {best['f05']:.4f}  (threshold {best['threshold']}, assign_best {best['assign_best']}) | "
          f"blocking ceiling {best['blocking_ceiling_f05']:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
