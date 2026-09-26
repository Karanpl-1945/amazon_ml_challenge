"""Pair features for every blocking candidate (input to the matcher).

For each split/country the candidate file work/cand/{split}_{country}.parquet is joined to the cleaned
records of both sides and scored with string similarities (rapidfuzz, vectorised C++), number / postcode
agreement, legal-suffix and alias checks, per-query and per-candidate competition statistics.
No country feature is used, so the model transfers to countries unseen in training (France).

Output: work/feat/{split}_{country}.parquet  (s1_id, cand_id, features..., [label, is_val] for train)
Train: only queries in the fit set (validation fold + 20% sample of the train fold) are featurised;
competition statistics still use every candidate row, so they match test-time conditions.

Run: python src/features.py --split train|test [--workers 8]
"""
import argparse
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

import config
from blocking import CAND_DIR, gt_pair_table

FEAT_DIR = config.WORK_DIR / "feat"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

ATTR = ["entity_id", "name_clean", "name_core", "name_alias", "name_legal", "addr_clean", "addr_nums",
        "addr_first_num", "addr_postcode", "addr_landmark", "addr_comps"]
FLAGS = ["f_name_domain", "f_name_alias", "f_name_native", "f_addr_native", "f_name_caps", "f_addr_caps",
         "f_addr_empty", "f_name_junk"]
PAIR_CHUNK = 500_000


# ----------------------------------------------------------------------------- competition statistics
def competition(cand):
    """Per-query and per-candidate context from the whole candidate table (numpy group-bys)."""
    s1 = pc.dictionary_encode(cand["s1_id"].combine_chunks()).indices.to_numpy()
    cd = pc.dictionary_encode(cand["cand_id"].combine_chunks()).indices.to_numpy()
    sc = cand["score"].to_numpy()
    cn = cand["cos_name"].to_numpy()
    ca = cand["cos_addr"].to_numpy()
    src3 = pc.starts_with(cand["cand_id"], "S3-").to_numpy(zero_copy_only=False)
    out = {}

    def grp_max(key, v):
        m = np.full(key.max() + 1, -np.inf, np.float32)
        np.maximum.at(m, key, v)
        return m[key]

    def grp_rank(key, v):
        """0-based descending rank of v inside each key group."""
        order = np.lexsort((-v, key))
        ks = key[order]
        start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
        pos = np.arange(len(ks)) - np.repeat(start, np.diff(np.r_[start, len(ks)]))
        r = np.empty(len(v), np.int32)
        r[order] = pos
        return r

    def grp_second(key, v, first):
        """Best value in the group excluding the row itself (second best for the argmax row)."""
        order = np.lexsort((-v, key))
        ks, vs = key[order], v[order]
        start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
        size = np.diff(np.r_[start, len(ks)])
        sec = np.full(key.max() + 1, 0.0, np.float32)
        has2 = size > 1
        sec[ks[start[has2]]] = vs[start[has2] + 1]
        s = sec[key]
        return np.where(v >= first, s, first)

    q_best = grp_max(s1, sc)
    out["q_best"] = q_best
    out["q_gap"] = q_best - sc
    out["q_n"] = np.bincount(s1)[s1].astype(np.int16)
    key_src = s1 * 2 + src3
    out["q_rank_src"] = grp_rank(key_src, sc).astype(np.int16)
    out["q_gap_src"] = grp_max(key_src, sc) - sc
    out["q_best_name"] = grp_max(s1, cn)
    out["q_best_addr"] = grp_max(s1, ca)
    c_best = grp_max(cd, sc)
    out["c_n"] = np.bincount(cd)[cd].astype(np.int16)
    out["c_rank"] = grp_rank(cd, sc).astype(np.int16)
    out["c_gap"] = c_best - sc
    out["c_margin"] = sc - grp_second(cd, sc, c_best)
    out["q_margin"] = sc - grp_second(s1, sc, q_best)
    return {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in out.items()}


# ----------------------------------------------------------------------------- string features
def _sim(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.uint8 if scorer is not JW else None)


JW = JaroWinkler.normalized_similarity


def _set_jaccard(a, b, sep=None):
    out = np.zeros(len(a), np.float32)
    both = np.zeros(len(a), np.int8)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            sx, sy = set(x.split(sep)), set(y.split(sep))
            out[i] = len(sx & sy) / len(sx | sy)
            both[i] = 1
    return out, both


def _agree(a, b):
    """1 = both present and equal, -1 = both present and different, 0 = at least one missing."""
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    pres = (a != "") & (b != "")
    return np.where(pres, np.where(a == b, 1, -1), 0).astype(np.int8)


def string_features(L, R, workers):
    """L, R: dict of python lists (left = Source 1, right = candidate)."""
    f = {}
    f["n_ratio"] = _sim(L["name_clean"], R["name_clean"], fuzz.ratio, workers)
    f["n_tset"] = _sim(L["name_clean"], R["name_clean"], fuzz.token_set_ratio, workers)
    f["nc_tsort"] = _sim(L["name_core"], R["name_core"], fuzz.token_sort_ratio, workers)
    f["nc_partial"] = _sim(L["name_core"], R["name_core"], fuzz.partial_ratio, workers)
    lc = [s.replace(" ", "") for s in L["name_core"]]
    rc = [s.replace(" ", "") for s in R["name_core"]]
    f["nc_jw"] = _sim(lc, rc, JW, workers).astype(np.float32)
    f["nc_compact_ratio"] = _sim(lc, rc, fuzz.ratio, workers)
    f["nc_equal"] = np.array([x == y and x != "" for x, y in zip(lc, rc)], np.int8)
    f["nc_jacc"], _ = _set_jaccard(L["name_core"], R["name_core"])
    f["n_tok_l"] = np.array([len(s.split()) for s in L["name_core"]], np.int8)
    f["n_tok_r"] = np.array([len(s.split()) for s in R["name_core"]], np.int8)
    f["n_len_diff"] = np.abs(np.array([len(s) for s in lc]) - np.array([len(s) for s in rc])).astype(np.int16)
    f["legal_agree"] = _agree(L["name_legal"], R["name_legal"])
    has_alias = np.array([bool(s) for s in R["name_alias"]])
    alias_sim = _sim(L["name_core"], R["name_alias"], fuzz.token_sort_ratio, workers)
    f["alias_sim"] = np.where(has_alias, alias_sim, 0).astype(np.uint8)
    f["a_ratio"] = _sim(L["addr_clean"], R["addr_clean"], fuzz.ratio, workers)
    f["a_tset"] = _sim(L["addr_clean"], R["addr_clean"], fuzz.token_set_ratio, workers)
    f["a_partial"] = _sim(L["addr_clean"], R["addr_clean"], fuzz.partial_token_set_ratio, workers)
    f["a_num_jacc"], f["a_num_both"] = _set_jaccard(L["addr_nums"], R["addr_nums"])
    f["a_comp_jacc"], _ = _set_jaccard(L["addr_comps"], R["addr_comps"], "|")
    f["a_first_num"] = _agree(L["addr_first_num"], R["addr_first_num"])
    f["a_postcode"] = _agree(L["addr_postcode"], R["addr_postcode"])
    f["lm_sim"] = np.where([bool(x) and bool(y) for x, y in zip(L["addr_landmark"], R["addr_landmark"])],
                           _sim(L["addr_landmark"], R["addr_landmark"], fuzz.token_set_ratio, workers), 255).astype(np.uint8)
    f["a_len_l"] = np.array([len(s) for s in L["addr_clean"]], np.int16)
    f["a_len_r"] = np.array([len(s) for s in R["addr_clean"]], np.int16)
    return f


# ----------------------------------------------------------------------------- driver
def _attrs(split, srcs, country, ids=None):
    """Cleaned attribute tables for `srcs`, filtered to country (and optionally to a set of ids)."""
    cols = ATTR + FLAGS
    parts = []
    for s in srcs:
        for b in pq.ParquetFile(config.CLEAN_DIR / f"{split}_{s}.parquet").iter_batches(
                batch_size=500_000, columns=cols + ["country"]):
            t = pa.Table.from_batches([b])
            m = pc.equal(t["country"], country)
            if ids is not None:
                m = pc.and_(m, pc.is_in(t["entity_id"], value_set=ids))
            parts.append(t.filter(m).drop_columns(["country"]))
    return pa.concat_tables(parts).combine_chunks()


def run_country(split, country, workers):
    t0 = time.time()
    tag = f"{split}_{country}"
    cand = pq.read_table(CAND_DIR / f"{tag}.parquet")
    comp = competition(cand)
    keep = None
    if split == "train":
        s1 = pq.read_table(config.CLEAN_DIR / "train_s1.parquet", columns=["entity_id", "is_val", "country"])
        s1 = s1.filter(pc.equal(s1["country"], country))
        ids, isv = s1["entity_id"].to_pylist(), s1["is_val"].to_numpy(zero_copy_only=False)
        fit = pa.array([e for e, v in zip(ids, isv) if v or config.is_fit_entity(e)])
        keep = pc.is_in(cand["s1_id"], value_set=fit).to_numpy(zero_copy_only=False)
        cand = cand.filter(pa.array(keep))
        comp = {k: v[keep] for k, v in comp.items()}
    left = _attrs(split, ["s1"], country, pc.unique(cand["s1_id"]))
    right = _attrs(split, ["s2", "s3"], country, pc.unique(cand["cand_id"]))
    li = pc.index_in(cand["s1_id"], value_set=left["entity_id"]).to_numpy()
    ri = pc.index_in(cand["cand_id"], value_set=right["entity_id"]).to_numpy()

    if split == "train":
        gt = gt_pair_table()
        lab = pc.is_in(pc.binary_join_element_wise(cand["s1_id"], cand["cand_id"], "|"),
                       value_set=pc.binary_join_element_wise(gt["s1_id"], gt["cand_id"], "|"))
        label = lab.to_numpy(zero_copy_only=False).astype(np.int8)
        s1v = pq.read_table(config.CLEAN_DIR / "train_s1.parquet", columns=["entity_id", "is_val"])
        s1v = s1v.filter(pc.is_in(s1v["entity_id"], value_set=left["entity_id"]))
        vi = pc.index_in(cand["s1_id"], value_set=s1v["entity_id"]).to_numpy()
        is_val = s1v["is_val"].to_numpy(zero_copy_only=False)[vi]

    writer = None
    n = cand.num_rows
    for s in range(0, n, PAIR_CHUNK):
        e = min(s + PAIR_CHUNK, n)
        lt = left.take(pa.array(li[s:e]))
        rt = right.take(pa.array(ri[s:e]))
        L = {c: lt[c].to_pylist() for c in ATTR[1:]}
        R = {c: rt[c].to_pylist() for c in ATTR[1:]}
        f = string_features(L, R, workers)
        cols = {"s1_id": cand["s1_id"].slice(s, e - s), "cand_id": cand["cand_id"].slice(s, e - s)}
        for c in ("cos_name", "cos_addr", "score", "rank"):
            cols[c] = cand[c].slice(s, e - s)
        cols["is_s3"] = pc.starts_with(cols["cand_id"], "S3-")
        for k, v in comp.items():
            cols[k] = v[s:e]
        for k, v in f.items():
            cols[k] = v
        for c in FLAGS:
            cols["r_" + c] = rt[c]
        cols["l_addr_empty"] = lt["f_addr_empty"]
        if split == "train":
            cols["label"] = label[s:e]
            cols["is_val"] = is_val[s:e]
        out = pa.table(cols)
        if writer is None:
            writer = pq.ParquetWriter(FEAT_DIR / f"{tag}.parquet", out.schema, compression="zstd")
        writer.write_table(out)
        print(f"    {tag}: {e:,}/{n:,} pairs  ({time.time()-t0:.0f}s)", flush=True)
    writer.close()
    print(f"  {tag}: {n:,} pairs featurised in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--country", default=None)
    args = ap.parse_args()
    countries = [args.country] if args.country else sorted(
        p.stem.split("_", 1)[1] for p in CAND_DIR.glob(f"{args.split}_*.parquet"))
    for c in countries:
        run_country(args.split, c, args.workers)


if __name__ == "__main__":
    sys.exit(main())
