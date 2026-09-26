"""Candidate generation (blocking): weighted rare-feature probing + TF-IDF cosine re-ranking.

Per country (an open set - whatever labels Source 1 carries), every record becomes a sparse set of
integer features:
  name group : name tokens, token prefix/suffix, space-less name + its prefix/suffix  (see _group_flat)
  addr group : address tokens (+ split compound numbers), house number, postcode
Feature weight w = idf^2 with idf = log(1 + N_target / df_target), so the record norm is a TF-IDF norm.

For each Source-1 record (query) numba does:
  1. probe : walk the posting lists of its `m_name` rarest name features and `m_addr` rarest address
             features (df <= max_df) and accumulate w into a dense per-thread score array
  2. rerank: the top `M` touched targets get the exact name cosine and address cosine
             (sorted-feature intersection); score = cos_name + ADDR_W * cos_addr
  3. keep  : the best `K` by score
Output per split/country: work/cand/{split}_{country}.parquet with s1_id, cand_id, cos_name, cos_addr,
score, rank (0 = best). Candidates for train come from the full S2+S3 pool, so they are realistic.

Run: python src/blocking.py --split train [--mode all|val]   (prints validation recall@k)
     python src/blocking.py --split test
"""
import argparse
import sys
import time

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numba import njit, prange

import config
from io_utils import load_gt_pairs

CAND_DIR = config.WORK_DIR / "cand"
CAND_DIR.mkdir(parents=True, exist_ok=True)

COLS = ["entity_id", "country", "name_core", "name_alias", "addr_clean", "addr_first_num", "addr_postcode"]
ADDR_W = 1.3
MAX_DF = 10000
M_NAME, M_ADDR = 5, 5
RERANK_M = 3000


# ----------------------------------------------------------------------------- featurisation (Arrow)
def _col(tbl, name):
    return tbl[name].combine_chunks()


def _flat(lists):
    """ListArray(s) -> (parent_record_idx int64, flat string array), empty strings dropped."""
    parents, vals = [], []
    for la in lists:
        parents.append(pc.list_parent_indices(la).to_numpy())
        vals.append(pc.list_flatten(la))
    par, val = np.concatenate(parents), pa.concat_arrays(vals)
    keep = pc.utf8_length(val).to_numpy() > 0
    return par[keep], val.filter(pa.array(keep))


def _single(arr):
    """String array -> (record idx, value) for the non-empty values."""
    keep = pc.utf8_length(arr).to_numpy() > 0
    return np.nonzero(keep)[0], arr.filter(pa.array(keep))


def _affixes(parent, vals, n, min_len, tag):
    """n-char prefix and suffix of values of length >= min_len (typo-tolerant features)."""
    keep = pc.utf8_length(vals).to_numpy() >= min_len
    v = vals.filter(pa.array(keep))
    p = parent[keep]
    pre = pc.binary_join_element_wise(tag + "<", pc.utf8_slice_codeunits(v, 0, n), "")
    suf = pc.binary_join_element_wise(tag + ">", pc.utf8_slice_codeunits(v, -n), "")
    return np.concatenate([p, p]), pa.concat_arrays([pre, suf])


def _group_flat(tbl):
    """-> list of (group, parent_idx, flat_values). Groups n/p/c are name features, a/f/z address.
    n: name tokens (core + alias)       p: 4-char prefix/suffix of name tokens >= 5 chars
    c: whole name_core with spaces removed + its 5-char prefix/suffix (catches 'dypharmaceuticals')
    a: address tokens, plus sub-tokens of compound numbers split on / and - with leading zeros dropped
    f: house number                      z: postcode"""
    ws = pc.utf8_split_whitespace
    out = []
    pn, vn = _flat([ws(_col(tbl, "name_core")), ws(_col(tbl, "name_alias"))])
    out.append(("n", pn, vn))
    out.append(("p", *_affixes(pn, vn, 4, 5, "")))
    pc_, vc = _single(pc.replace_substring(_col(tbl, "name_core"), " ", ""))
    pa_, va = _affixes(pc_, vc, 5, 7, "c")
    out.append(("c", np.concatenate([pc_, pa_]), pa.concat_arrays([vc, va])))
    pa1, va1 = _flat([ws(_col(tbl, "addr_clean"))])
    cmp_ = pc.match_substring_regex(va1, r"[/\-]|^0").to_numpy(zero_copy_only=False)
    parts = pc.split_pattern_regex(va1.filter(pa.array(cmp_)), r"[/\-]")
    pa2 = pa1[cmp_][pc.list_parent_indices(parts).to_numpy()]
    va2 = pc.utf8_ltrim(pc.list_flatten(parts), "0")
    nz = pc.utf8_length(va2).to_numpy() > 0
    pa2, va2 = pa2[nz], va2.filter(pa.array(nz))
    out.append(("a", np.concatenate([pa1, pa2]), pa.concat_arrays([va1, va2])))
    out.append(("f", *_single(pc.utf8_ltrim(_col(tbl, "addr_first_num"), "0"))))
    out.append(("z", *_single(_col(tbl, "addr_postcode"))))
    return out


NAME_GROUPS = ("n", "p", "c")


class Vocab:
    """Per-group string vocabularies built on the target side; global id = group offset + local id."""

    def __init__(self):
        self.values = {}

    def update(self, tbl):
        for g, _, v in _group_flat(tbl):
            u = pc.unique(v)
            self.values[g] = u if g not in self.values else pc.unique(pa.concat_arrays([self.values[g], u]))

    def freeze(self):
        self.offset, off = {}, 0
        for g, _, _ in _group_flat(_EMPTY):
            self.offset[g] = off
            off += len(self.values.get(g, []))
            if g in NAME_GROUPS:
                self.n_name = off
        self.n_feat = off

    def encode(self, tbl):
        """-> CSR (ptr int64, feat int32) of sorted unique global feature ids; unseen strings dropped."""
        pars, fids = [], []
        for g, p, v in _group_flat(tbl):
            if g not in self.values:
                continue
            idx = pc.index_in(v, value_set=self.values[g])
            ok = idx.is_valid().to_numpy(zero_copy_only=False)
            pars.append(p[ok].astype(np.int64))
            fids.append((idx.fill_null(0).to_numpy()[ok] + self.offset[g]).astype(np.int32))
        return _csr_rows(np.concatenate(pars), np.concatenate(fids), tbl.num_rows)


_EMPTY = pa.table({c: pa.array([], pa.string()) for c in COLS})


def _batches(path, country, extra=(), batch=400_000):
    """Stream one cleaned parquet file, keeping only rows of `country`."""
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=COLS + list(extra)):
        t = pa.Table.from_batches([b])
        yield t.filter(pc.equal(t["country"], country))


def _encode_stream(vocab, tables):
    """Encode a stream of tables into one CSR + concatenated entity ids (memory: ints only)."""
    ptrs, feats, ids, base = [np.zeros(1, np.int64)], [], [], 0
    for t in tables:
        if t.num_rows == 0:
            continue
        p, f = vocab.encode(t)
        ptrs.append(p[1:] + base)
        feats.append(f)
        base += len(f)
        ids.append(t["entity_id"].combine_chunks())
    return np.concatenate(ptrs), np.concatenate(feats), pa.concat_arrays(ids)


# ----------------------------------------------------------------------------- numba kernels
@njit(cache=True)
def _csr_rows(parent, fid, n_rec):
    """Group (parent, feature) pairs into per-record sorted unique feature lists."""
    cnt = np.zeros(n_rec + 1, np.int64)
    for r in parent:
        cnt[r + 1] += 1
    ptr = np.cumsum(cnt)
    pos = ptr[:-1].copy()
    tmp = np.empty(len(fid), np.int32)
    for i in range(len(fid)):
        r = parent[i]
        tmp[pos[r]] = fid[i]
        pos[r] += 1
    out_ptr = np.zeros(n_rec + 1, np.int64)
    out = np.empty(len(fid), np.int32)
    k = 0
    for r in range(n_rec):
        row = np.sort(tmp[ptr[r]:ptr[r + 1]])
        prev = -1
        for f in row:
            if f != prev:
                out[k] = f
                k += 1
                prev = f
        out_ptr[r + 1] = k
    return out_ptr, out[:k]


@njit(cache=True)
def _postings(d_ptr, d_feat, n_feat):
    cnt = np.zeros(n_feat + 1, np.int64)
    for f in d_feat:
        cnt[f + 1] += 1
    fptr = np.cumsum(cnt)
    pos = fptr[:-1].copy()
    post = np.empty(len(d_feat), np.int32)
    for r in range(len(d_ptr) - 1):
        for j in range(d_ptr[r], d_ptr[r + 1]):
            f = d_feat[j]
            post[pos[f]] = r
            pos[f] += 1
    return fptr, post


@njit(cache=True)
def _norms(ptr, feat, w, n_name):
    n = len(ptr) - 1
    nn = np.zeros(n, np.float32)
    na = np.zeros(n, np.float32)
    for r in range(n):
        for j in range(ptr[r], ptr[r + 1]):
            f = feat[j]
            if f < n_name:
                nn[r] += w[f]
            else:
                na[r] += w[f]
    return np.sqrt(nn), np.sqrt(na)


@njit(parallel=True, cache=True)
def _block(q_ptr, q_feat, q_nn, q_na, d_ptr, d_feat, d_nn, d_na, fptr, post, df, w, n_name,
           max_df, m_name, m_addr, rerank_m, k, addr_w):
    nq = len(q_ptr) - 1
    nd = len(d_ptr) - 1
    out_d = np.full((nq, k), -1, np.int32)
    out_cn = np.zeros((nq, k), np.float32)
    out_ca = np.zeros((nq, k), np.float32)
    n_chunks = 256
    for c in prange(n_chunks):
        acc = np.zeros(nd, np.float32)
        touched = np.empty((m_name + m_addr) * max_df, np.int32)
        for q in range(c * nq // n_chunks, (c + 1) * nq // n_chunks):
            a, b = q_ptr[q], q_ptr[q + 1]
            if b == a:
                continue
            qf = q_feat[a:b]
            # probe features: rarest name and rarest address features within max_df
            qdf = df[qf]
            order = np.argsort(qdf)
            nt = 0
            used_n = 0
            used_a = 0
            for oi in order:
                f = qf[oi]
                dff = qdf[oi]
                if dff == 0 or dff > max_df:
                    continue
                if f < n_name:
                    if used_n >= m_name:
                        continue
                    used_n += 1
                else:
                    if used_a >= m_addr:
                        continue
                    used_a += 1
                wf = w[f]
                for j in range(fptr[f], fptr[f + 1]):
                    d = post[j]
                    if acc[d] == 0.0:
                        touched[nt] = d
                        nt += 1
                    acc[d] += wf
            if nt == 0:
                continue
            tl = touched[:nt].copy()
            vals = np.empty(nt, np.float32)
            for i in range(nt):
                vals[i] = -acc[tl[i]]
                acc[tl[i]] = 0.0
            sel = np.argsort(vals)[:rerank_m]
            ns = len(sel)
            sc = np.empty(ns, np.float32)
            cn = np.empty(ns, np.float32)
            ca = np.empty(ns, np.float32)
            for si in range(ns):
                d = tl[sel[si]]
                i, j = a, d_ptr[d]
                je = d_ptr[d + 1]
                sn = 0.0
                sa = 0.0
                while i < b and j < je:
                    fi = q_feat[i]
                    fj = d_feat[j]
                    if fi == fj:
                        if fi < n_name:
                            sn += w[fi]
                        else:
                            sa += w[fi]
                        i += 1
                        j += 1
                    elif fi < fj:
                        i += 1
                    else:
                        j += 1
                x = q_nn[q] * d_nn[d]
                y = q_na[q] * d_na[d]
                cn[si] = sn / x if x > 0 else 0.0
                ca[si] = sa / y if y > 0 else 0.0
                sc[si] = -(cn[si] + addr_w * ca[si])
            best = np.argsort(sc)[:k]
            for r in range(len(best)):
                bi = best[r]
                out_d[q, r] = tl[sel[bi]]
                out_cn[q, r] = cn[bi]
                out_ca[q, r] = ca[bi]
    return out_d, out_cn, out_ca


# ----------------------------------------------------------------------------- driver
def _rss():
    return f"{psutil.Process().memory_info().rss / 1e9:.2f}GB"


Q_CHUNK = 150_000


def _query_filter(split, mode):
    """Row filter for Source-1 queries. test: all. train: val -> validation fold only;
    fit -> validation fold + a 20% sample of the train fold (model training data); all -> everything."""
    if split == "test" or mode == "all":
        return None
    if mode == "val":
        return lambda ids, is_val: is_val
    return lambda ids, is_val: is_val | np.array([config.is_fit_entity(e) for e in ids.to_pylist()])


def run_country(split, country, k, mode):
    t0 = time.time()
    clean = config.CLEAN_DIR
    d_files = [clean / f"{split}_{s}.parquet" for s in ("s2", "s3")]
    vocab = Vocab()
    for f in d_files:
        for t in _batches(f, country):
            vocab.update(t)
    vocab.freeze()
    pa.default_memory_pool().release_unused()
    print(f"    vocab built ({time.time()-t0:.0f}s, rss {_rss()})", flush=True)
    d_ptr, d_feat, d_ids = _encode_stream(vocab, (t for f in d_files for t in _batches(f, country)))

    extra = ("is_val",) if split == "train" else ()
    keep = _query_filter(split, mode)

    def q_tables():
        for t in _batches(clean / f"{split}_s1.parquet", country, extra):
            if keep is not None and t.num_rows:
                t = t.filter(pa.array(keep(t["entity_id"], t["is_val"].to_numpy(zero_copy_only=False))))
            yield t
    q_ptr, q_feat, q_ids = _encode_stream(vocab, q_tables())

    n_feat, n_name, nd = vocab.n_feat, vocab.n_name, len(d_ids)
    del vocab
    df = np.bincount(d_feat, minlength=n_feat).astype(np.int32)
    idf = np.log1p(nd / np.maximum(df, 1)).astype(np.float32)
    w = idf * idf
    fptr, post = _postings(d_ptr, d_feat, n_feat)
    q_nn, q_na = _norms(q_ptr, q_feat, w, n_name)
    d_nn, d_na = _norms(d_ptr, d_feat, w, n_name)
    pa.default_memory_pool().release_unused()
    print(f"    index built ({time.time()-t0:.0f}s, rss {_rss()})", flush=True)
    t1 = time.time()

    tag = f"{split}_{country}"
    writer, n_pairs = None, 0
    nq = len(q_ids)
    for s in range(0, nq, Q_CHUNK):
        e = min(s + Q_CHUNK, nq)
        sub_ptr = q_ptr[s:e + 1] - q_ptr[s]
        sub_feat = q_feat[q_ptr[s]:q_ptr[e]]
        out_d, out_cn, out_ca = _block(sub_ptr, sub_feat, q_nn[s:e], q_na[s:e], d_ptr, d_feat, d_nn, d_na,
                                       fptr, post, df, w, n_name, MAX_DF, M_NAME, M_ADDR, RERANK_M, k, ADDR_W)
        valid = out_d >= 0
        qi, rk = np.nonzero(valid)
        cn, ca = out_cn[valid], out_ca[valid]
        res = pa.table({"s1_id": q_ids.take(pa.array(qi + s)), "cand_id": d_ids.take(pa.array(out_d[valid])),
                        "cos_name": cn, "cos_addr": ca, "score": cn + ADDR_W * ca, "rank": rk.astype(np.int16)})
        if writer is None:
            writer = pq.ParquetWriter(CAND_DIR / f"{tag}.parquet", res.schema, compression="zstd")
        writer.write_table(res)
        n_pairs += res.num_rows
    if writer is not None:
        writer.close()
    print(f"  {tag}: {nq:,} queries x {nd:,} targets, {n_feat:,} feats, {len(d_feat):,} postings | "
          f"featurize {t1-t0:.0f}s, block {time.time()-t1:.0f}s | {n_pairs:,} pairs", flush=True)
    return tag


def gt_pair_table():
    """Ground truth exploded to an Arrow table (s1_id, cand_id)."""
    gt = pq.read_table(config.TRAIN_GT) if str(config.TRAIN_GT).endswith(".parquet") else None
    if gt is None:
        from io_utils import read_tsv_table
        gt = read_tsv_table(config.TRAIN_GT)
    lists = pc.split_pattern(gt["matched_entity_ids"].combine_chunks(), ",")
    s1 = gt["source1_entity_id"].combine_chunks().take(pc.list_parent_indices(lists))
    m = pc.list_flatten(lists)
    t = pa.table({"s1_id": s1, "cand_id": m})
    return t.filter(pc.greater(pc.utf8_length(t["cand_id"]), 0))


def recall_report(tags, ks=(1, 3, 5, 10, 15, 20, 30, 40), val_only=True):
    cand = pa.concat_tables([pq.read_table(CAND_DIR / f"{t}.parquet", columns=["s1_id", "cand_id", "rank"])
                             for t in tags])
    s1 = pq.read_table(config.CLEAN_DIR / "train_s1.parquet", columns=["entity_id", "is_val"])
    if val_only:
        s1 = s1.filter(s1["is_val"])
    queries = pc.unique(cand["s1_id"])
    queries = pc.filter(queries, pc.is_in(queries, value_set=s1["entity_id"].combine_chunks()))
    truth = gt_pair_table()
    truth = truth.filter(pc.is_in(truth["s1_id"], value_set=queries))
    hit = truth.join(cand, keys=["s1_id", "cand_id"], join_type="inner")
    r = hit["rank"].to_numpy()
    n = truth.num_rows
    print(f"{len(queries):,} {'val ' if val_only else ''}queries, {n:,} true pairs")
    for kk in ks:
        print(f"  recall@{kk:<3} {np.sum(r < kk) / n:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--mode", choices=["val", "fit", "all"], default="all",
                    help="train queries: all (pipeline default), val = validation fold only (fast recall check)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--country", default=None)
    ap.add_argument("--rerank", type=int, default=RERANK_M)
    args = ap.parse_args()
    globals()["RERANK_M"] = args.rerank
    countries = [args.country] if args.country else sorted(
        pc.unique(pq.read_table(config.CLEAN_DIR / f"{args.split}_s1.parquet", columns=["country"])["country"]).to_pylist())
    t0 = time.time()
    tags = [run_country(args.split, c, args.k, args.mode) for c in countries]
    print(f"blocking done in {time.time()-t0:.0f}s")
    if args.split == "train":
        recall_report(tags)


if __name__ == "__main__":
    sys.exit(main())
