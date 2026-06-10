"""
subsample.py
────────────
Build a tractable evaluation set from the huge labeled parquet WITHOUT loading
it all into RAM. Keeps ALL positives; randomly samples negatives.

Streams the parquet in row-group batches (pyarrow), so peak memory stays small.

Usage:
  python subsample.py --in out/labeled.parquet --out out/labeled_small.parquet --neg 2000000
"""

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
import argparse, os


def main(in_path, out_path, n_neg, seed):
    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(in_path)
    n_rowgroups = pf.num_row_groups
    print(f"[subsample] {in_path}: {pf.metadata.num_rows:,} rows, "
          f"{n_rowgroups} row groups")

    pos_parts = []          # keep ALL positives
    neg_parts = []          # sampled negatives
    total_pos = 0
    total_neg_seen = 0

    # first pass cheap: get total negative count to set a per-row sampling rate
    total_neg = pf.metadata.num_rows  # upper bound; refine with label if cheap
    # we don't know exact neg count without scanning, so use num_rows as ~neg
    frac = min(1.0, n_neg / max(total_neg, 1)) * 1.15  # slight over-sample buffer

    for rg in range(n_rowgroups):
        batch = pf.read_row_group(rg).to_pandas()
        pos = batch[batch["label"] == 1]
        if len(pos):
            pos_parts.append(pos)
            total_pos += len(pos)

        neg = batch[batch["label"] == 0]
        total_neg_seen += len(neg)
        if len(neg):
            take = neg.sample(frac=min(frac, 1.0), random_state=seed + rg)
            neg_parts.append(take)
        if rg % 5 == 0:
            cur_neg = sum(len(x) for x in neg_parts)
            print(f"  row group {rg}/{n_rowgroups}: pos so far={total_pos}, "
                  f"neg sampled={cur_neg:,}")

    pos_df = pd.concat(pos_parts, ignore_index=True) if pos_parts else pd.DataFrame()
    neg_df = pd.concat(neg_parts, ignore_index=True) if neg_parts else pd.DataFrame()
    # trim negatives to exactly n_neg if we over-sampled
    if len(neg_df) > n_neg:
        neg_df = neg_df.sample(n=n_neg, random_state=seed).reset_index(drop=True)
    small = pd.concat([pos_df, neg_df], ignore_index=True).sort_values(
        "time").reset_index(drop=True)
    small.to_parquet(out_path, index=False)
    print(f"[done] saved {out_path}: {small.shape}, "
          f"positives={int(small['label'].sum())}, "
          f"negatives={len(small) - int(small['label'].sum()):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="out/labeled.parquet")
    ap.add_argument("--out", default="out/labeled_small.parquet")
    ap.add_argument("--neg", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.inp, a.out, a.neg, a.seed)
