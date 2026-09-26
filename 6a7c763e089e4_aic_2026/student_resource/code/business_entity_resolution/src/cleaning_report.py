"""Measure how much cleaning helps, on VALIDATION-fold true pairs (unseen by the dictionary miner).

For a sample of true (S1, S2/S3) pairs, compares similarity of raw-lowercased fields vs cleaned
fields, split by country / source / native-script flag. Higher similarity for true pairs (while
random pairs stay low) = easier blocking and matching.

Run (after clean_data.py):  python src/cleaning_report.py [--n 200000]
"""
import argparse

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel

import config
from io_utils import load_gt_pairs


def _ratio(a, b):
    return np.array([Indel.normalized_similarity(x, y) for x, y in zip(a, b)])


def _tok_set(a, b):
    return np.array([fuzz.token_set_ratio(x, y) / 100 for x, y in zip(a, b)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    args = ap.parse_args()

    pairs, _ = load_gt_pairs(config.TRAIN_GT)
    val = pd.DataFrame([p for p in pairs if config.is_val_entity(p[0])], columns=["s1_id", "m_id"])
    val = val.sample(min(args.n, len(val)), random_state=config.SEED)

    cols = ["entity_id", "business_name", "business_address", "country", "name_clean", "name_core",
            "addr_clean", "addr_first_num", "f_name_native", "f_addr_native"]
    s1 = pd.read_parquet(config.CLEAN_DIR / "train_s1.parquet", columns=cols)
    s1 = s1[s1.entity_id.isin(val.s1_id)]
    s23 = pd.concat([pd.read_parquet(config.CLEAN_DIR / f"train_{s}.parquet", columns=cols,
                                     filters=[("entity_id", "in", val.m_id.tolist())]) for s in ("s2", "s3")])
    df = val.merge(s1, left_on="s1_id", right_on="entity_id").merge(s23, left_on="m_id", right_on="entity_id", suffixes=("", "_m"))

    # random same-country negatives for contrast: shuffle the S2/S3 side within each country
    df = df.sort_values("country").reset_index(drop=True)
    neg = df.copy()
    rng = np.random.default_rng(0)
    m_cols = [c for c in df.columns if c.endswith("_m")] + ["m_id"]
    for _, idx in df.groupby("country").indices.items():
        neg.loc[idx, m_cols] = df.loc[rng.permutation(idx), m_cols].to_numpy()

    out = []
    for label, d in (("true", df), ("random", neg)):
        r = {
            "raw name ratio": _ratio(d.business_name.str.lower(), d.business_name_m.str.lower()),
            "clean name ratio": _ratio(d.name_clean, d.name_clean_m),
            "core name ratio": _ratio(d.name_core, d.name_core_m),
            "core name exact": (d.name_core.to_numpy() == d.name_core_m.to_numpy()).astype(float),
            "raw addr token_set": _tok_set(d.business_address.str.lower(), d.business_address_m.str.lower()),
            "clean addr token_set": _tok_set(d.addr_clean, d.addr_clean_m),
            "same first number": ((d.addr_first_num.to_numpy() == d.addr_first_num_m.to_numpy())
                                  & (d.addr_first_num.to_numpy() != "")).astype(float),
        }
        m = pd.DataFrame(r)
        m["label"], m["country"], m["src"] = label, d.country.to_numpy(), d.m_id.str[:2].to_numpy()
        m["native"] = d.f_name_native_m.to_numpy()
        out.append(m)
    res = pd.concat(out)
    metrics = [c for c in res.columns if c not in ("label", "country", "src", "native")]
    pd.set_option("display.width", 250)
    print("=== overall (mean) ===")
    print(res.groupby("label")[metrics].mean().T.round(3))
    print("\n=== true pairs by country / source ===")
    print(res[res.label == "true"].groupby(["country", "src"])[metrics].mean().T.round(3))
    print("\n=== true pairs: native-script noisy name vs not ===")
    print(res[res.label == "true"].groupby("native")[metrics].mean().T.round(3))


if __name__ == "__main__":
    main()
