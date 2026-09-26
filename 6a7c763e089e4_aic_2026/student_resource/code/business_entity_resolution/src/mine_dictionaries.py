"""Mine noisy-form -> canonical-form dictionaries from labelled training pairs.

Source 1 is the clean reference, so for every true (S1, S2/S3) pair we align the two token
sequences and count which noisy token/component corresponds to which S1 token/component:

  * exact common tokens are matched first (they vote for the identity mapping),
  * the remaining tokens are aligned by position when both residuals have the same length.

A noisy form b is mapped to canonical a when  count(b->a) >= MIN_COUNT  and
count(b->a) / count(b aligned to anything, identity included) >= MIN_SHARE. Identity votes keep
genuine words (e.g. "road") from being remapped by occasional misalignments.

Only the train fold (config.is_val_entity == False) is used so validation remains unbiased.

Outputs (in config.DICT_DIR):
  name_token_map.json, addr_token_map.json, addr_component_map.json   {country|_all: {noisy: canon}}
  mining_report.json   (sizes + top name insertions, for inspection)

Run:  python src/mine_dictionaries.py [--max-pairs 3000000] [--workers 12]
"""
import argparse
import json
import random
import time
from collections import Counter
from multiprocessing import Pool

import pandas as pd

import config
from io_utils import load_gt_pairs, load_tsv
from rapidfuzz import fuzz

from normalize import base_components, base_name_tokens

MIN_COUNT = {"name": 3, "addr": 4, "comp": 5}
MIN_SHARE = {"name": 0.5, "addr": 0.5, "comp": 0.5}
MAX_RESIDUAL = 6


def _align(A, B, cnt, tot, ctry):
    """Update counters with the alignment of noisy sequence B onto canonical sequence A.
    Returns the residual (unmatched) lists."""
    ca = Counter(A)
    ra, rb = [], []
    for b in B:
        if ca.get(b, 0) > 0:
            ca[b] -= 1
            cnt[(ctry, b, b)] += 1
            tot[(ctry, b)] += 1
        else:
            rb.append(b)
    cb = Counter(B)
    for a in A:
        if cb.get(a, 0) > 0:
            cb[a] -= 1
        else:
            ra.append(a)
    if ra and len(ra) == len(rb) and len(ra) <= MAX_RESIDUAL:
        for a, b in zip(ra, rb):
            cnt[(ctry, b, a)] += 1
            tot[(ctry, b)] += 1
    return ra, rb


def _mine_chunk(rows):
    """Worker: rows = list of (country, name1, addr1, name2, addr2)."""
    c = {k: Counter() for k in ("name_cnt", "name_tot", "addr_cnt", "addr_tot", "comp_cnt", "comp_tot", "ins")}
    for ctry, n1, a1, n2, a2 in rows:
        ctry = ctry.strip().lower()
        A, B = base_name_tokens(n1)[0], base_name_tokens(n2)[0]
        ra, rb = _align(A, B, c["name_cnt"], c["name_tot"], ctry)
        if not ra and len(rb) == 1:                      # pure insertion -> noise word statistics
            pos = "first" if B and B[0] == rb[0] else ("last" if B[-1] == rb[0] else "mid")
            c["ins"][(pos, rb[0])] += 1
        if a2.strip():
            CA, CB = base_components(a1), base_components(a2)
            _align([" ".join(x) for x in CA], [" ".join(x) for x in CB], c["comp_cnt"], c["comp_tot"], ctry)
            _align([t for x in CA for t in x], [t for x in CB for t in x], c["addr_cnt"], c["addr_tot"], ctry)
    return c


def _is_subsequence(short, long):
    it = iter(long)
    return all(ch in it for ch in short)


def _plausible(b, a):
    """Reject misalignments between unrelated ASCII strings (e.g. a city aligned to a state).
    Accept typo-like pairs, containment (613-a -> 613) and abbreviations (tx -> texas, mh -> maharashtra).
    Non-ASCII (native script) keys are always accepted - they cannot be compared by spelling."""
    if not (b.isascii() and a.isascii()) or b == "#":
        return True
    short, long = sorted((b.replace(" ", ""), a.replace(" ", "")), key=len)
    if not short:
        return False
    if short in long or fuzz.ratio(a, b) >= 70:
        return True
    return short[0] == long[0] and _is_subsequence(short, long)


def _build_map(cnt, tot, kind):
    """Counters -> {country: {noisy: canonical}} plus a pooled '_all' map."""
    pooled_cnt, pooled_tot = Counter(), Counter()
    for (ctry, b, a), v in cnt.items():
        pooled_cnt[("_all", b, a)] += v
    for (ctry, b), v in tot.items():
        pooled_tot[("_all", b)] += v
    best = {}
    for source_cnt, source_tot in ((cnt, tot), (pooled_cnt, pooled_tot)):
        for (ctry, b, a), v in source_cnt.items():
            if a == b or v < MIN_COUNT[kind] or v / source_tot[(ctry, b)] < MIN_SHARE[kind]:
                continue
            if kind == "name" and b.isdigit():
                continue
            if not _plausible(b, a):
                continue
            key = (ctry, b)
            if key not in best or v > best[key][1]:
                best[key] = (a, v)
    out = {}
    for (ctry, b), (a, _) in best.items():
        out.setdefault(ctry, {})[b] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=3_000_000)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()

    pairs, _ = load_gt_pairs(config.TRAIN_GT)
    pairs = [p for p in pairs if not config.is_val_entity(p[0])]
    random.Random(config.SEED).shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    print(f"train-fold pairs used: {len(pairs):,}  ({time.time()-t0:.0f}s)")

    pdf = pd.DataFrame(pairs, columns=["s1_id", "m_id"])
    s1 = load_tsv(config.TRAIN_FILES["s1"]).rename(columns=lambda c: c + "_1")
    pdf = pdf.merge(s1, left_on="s1_id", right_on="entity_id_1")
    del s1
    s23 = pd.concat([load_tsv(config.TRAIN_FILES["s2"]), load_tsv(config.TRAIN_FILES["s3"])], ignore_index=True)
    s23 = s23[s23.entity_id.isin(pdf.m_id)]
    pdf = pdf.merge(s23, left_on="m_id", right_on="entity_id")
    del s23
    rows = list(zip(pdf.country_1.astype(str), pdf.business_name_1.astype(str), pdf.business_address_1.astype(str),
                    pdf.business_name.astype(str), pdf.business_address.astype(str)))
    del pdf
    print(f"joined {len(rows):,} pairs  ({time.time()-t0:.0f}s)")

    chunk = 50_000
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    total = {k: Counter() for k in ("name_cnt", "name_tot", "addr_cnt", "addr_tot", "comp_cnt", "comp_tot", "ins")}
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_mine_chunk, chunks)):
            for k in total:
                total[k].update(res[k])
            if i % 10 == 0:
                print(f"  chunk {i+1}/{len(chunks)}  ({time.time()-t0:.0f}s)")

    maps = {
        "name_token_map": _build_map(total["name_cnt"], total["name_tot"], "name"),
        "addr_token_map": _build_map(total["addr_cnt"], total["addr_tot"], "addr"),
        "addr_component_map": _build_map(total["comp_cnt"], total["comp_tot"], "comp"),
    }
    for name, m in maps.items():
        (config.DICT_DIR / f"{name}.json").write_text(json.dumps(m, ensure_ascii=False, indent=0), "utf-8")
    report = {
        "pairs": len(rows),
        "sizes": {name: {c: len(v) for c, v in m.items()} for name, m in maps.items()},
        "top_name_insertions": [(pos, tok, n) for (pos, tok), n in total["ins"].most_common(80)],
    }
    (config.DICT_DIR / "mining_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(report["sizes"], indent=1))
    print(f"done in {time.time()-t0:.0f}s -> {config.DICT_DIR}")


if __name__ == "__main__":
    main()
