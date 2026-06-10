"""
robustness.py
-------------
Repeat the full evaluation over N random negative-subsamples (seeds) and report
mean ± std for every headline metric — PTG and all baselines. Answers the
"is this result luck?" question that reviewers always ask.

Inputs:
- The FULL labeled parquet (all positives + all negatives), so each seed draws
  a fresh negative sample.
- If that is too big to reload per seed, point it at labeled_small and it will
  reshuffle which negatives are kept by re-sampling from it. Less ideal, but it
  still shows variance.

Usage:
python robustness.py --full outputs/data/labeled.parquet --neg 500000 --seeds 5
# or, if full is impractical:
python robustness.py --small outputs/data/labeled_small.parquet --seeds 5
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(__file__))
CORE = os.path.join(ROOT, "core")
EXP = os.path.join(ROOT, "experiments")

if CORE not in sys.path:
    sys.path.insert(0, CORE)

if EXP not in sys.path:
    sys.path.append(EXP)

from ptg_evaluate import evaluate as ptg_evaluate
from baselines_fair import run_baselines


def subsample_from_full(full_path, n_neg, seed):
    """Stream the full parquet, keep all positives, sample n_neg negatives."""
    pf = pq.ParquetFile(full_path)

    pos_parts = []
    neg_parts = []

    total_neg = pf.metadata.num_rows
    frac = min(1.0, (n_neg / max(total_neg, 1)) * 1.15)

    for rg in range(pf.num_row_groups):
        batch = pf.read_row_group(rg).to_pandas()

        pos = batch[batch.label == 1]
        if len(pos):
            pos_parts.append(pos)

        neg = batch[batch.label == 0]
        if len(neg):
            neg_parts.append(neg.sample(frac=min(frac, 1.0), random_state=seed + rg))

    pos_df = pd.concat(pos_parts, ignore_index=True) if pos_parts else pd.DataFrame()
    neg_df = pd.concat(neg_parts, ignore_index=True) if neg_parts else pd.DataFrame()

    if len(neg_df) > n_neg:
        neg_df = neg_df.sample(n=n_neg, random_state=seed)

    return pd.concat([pos_df, neg_df], ignore_index=True).sort_values(
        "time"
    ).reset_index(drop=True)


def resample_small(small_path, seed):
    """Reshuffle negatives within an existing small set (variance proxy)."""
    df = pd.read_parquet(small_path)
    pos = df[df.label == 1]
    neg = df[df.label == 0].sample(frac=0.8, random_state=seed)

    return pd.concat([pos, neg], ignore_index=True).sort_values(
        "time"
    ).reset_index(drop=True)


def summarize(runs, key_path):
    vals = []

    for run in runs:
        value = run
        for key in key_path:
            value = value.get(key, {}) if isinstance(value, dict) else None
            if value is None:
                break

        if isinstance(value, (int, float)):
            vals.append(value)

    if not vals:
        return None

    return {
        "mean": round(float(np.mean(vals)), 5),
        "std": round(float(np.std(vals)), 5),
        "n": len(vals),
        "values": [round(x, 5) for x in vals],
    }


def main(full, small, n_neg, seeds, delta, alpha, beta, gamma, max_depth):
    ptg_runs = []
    baseline_runs = []

    for i in range(seeds):
        seed = 42 + i
        print(f"\n{'=' * 50}\n SEED {seed} ({i + 1}/{seeds})\n{'=' * 50}")

        if full:
            df = subsample_from_full(full, n_neg, seed)
        else:
            df = resample_small(small, seed)

        print(f" {len(df):,} events, {int(df.label.sum())} malicious")

        ptg_result, _ = ptg_evaluate(
            df,
            delta=delta,
            max_depth=max_depth,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            verbose=False,
        )
        ptg_runs.append(ptg_result)

        print(
            f" PTG PR-AUC={ptg_result['PR_AUC']} "
            f"P@25={ptg_result['det_at_budget'][25]['precision']} "
            f"P@100={ptg_result['det_at_budget'][100]['precision']}"
        )

        baseline_result = run_baselines(df, delta=delta, seed=seed)
        baseline_runs.append(baseline_result)

        print(
            f" IF PR-AUC={baseline_result['IsolationForest']['PR_AUC']} "
            f"OCSVM={baseline_result['OneClassSVM']['PR_AUC']} "
            f"LR={baseline_result.get('LogisticRegression_temporal', {}).get('PR_AUC', 'NA')}"
        )

    summary = {
        "config": {
            "seeds": seeds,
            "delta": delta,
            "weights": [alpha, beta, gamma],
            "max_depth": max_depth,
            "n_neg": n_neg if full else "resampled_small",
        },
        "PTG": {
            "PR_AUC": summarize(ptg_runs, ["PR_AUC"]),
            "ROC_AUC": summarize(ptg_runs, ["ROC_AUC"]),
            "precision@25": summarize(ptg_runs, ["det_at_budget", 25, "precision"]),
            "precision@100": summarize(ptg_runs, ["det_at_budget", 100, "precision"]),
            "DR@200": summarize(ptg_runs, ["det_at_budget", 200, "detection_rate"]),
            "recall@FPR0.001": summarize(ptg_runs, ["recall_at_fpr", 0.001]),
        },
        "IsolationForest": {
            "PR_AUC": summarize(baseline_runs, ["IsolationForest", "PR_AUC"]),
        },
        "OneClassSVM": {
            "PR_AUC": summarize(baseline_runs, ["OneClassSVM", "PR_AUC"]),
        },
        "LogisticRegression_supervised": {
            "PR_AUC": summarize(baseline_runs, ["LogisticRegression_temporal", "PR_AUC"]),
        },
    }

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/robustness.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 50}\n ROBUSTNESS SUMMARY (mean ± std over {seeds} seeds)\n{'=' * 50}")

    ptg_summary = summary["PTG"]
    print(f" PTG PR-AUC : {ptg_summary['PR_AUC']['mean']} ± {ptg_summary['PR_AUC']['std']}")
    print(f" PTG P@25 : {ptg_summary['precision@25']['mean']} ± {ptg_summary['precision@25']['std']}")
    print(f" PTG P@100 : {ptg_summary['precision@100']['mean']} ± {ptg_summary['precision@100']['std']}")
    print(
        f" IF PR-AUC : {summary['IsolationForest']['PR_AUC']['mean']} "
        f"± {summary['IsolationForest']['PR_AUC']['std']}"
    )
    print(
        f" OCSVM PR-AUC : {summary['OneClassSVM']['PR_AUC']['mean']} "
        f"± {summary['OneClassSVM']['PR_AUC']['std']}"
    )

    lr_summary = summary["LogisticRegression_supervised"]["PR_AUC"]
    if lr_summary:
        print(f" LR(sup) PR-AUC : {lr_summary['mean']} ± {lr_summary['std']}")

    print("\n[saved] outputs/results/robustness.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=None, help="full labeled parquet")
    ap.add_argument("--small", default="outputs/data/labeled_small.parquet")
    ap.add_argument("--neg", type=int, default=500000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--delta", type=int, default=3600)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.2)
    a = ap.parse_args()

    main(
        a.full,
        a.small,
        a.neg,
        a.seeds,
        a.delta,
        a.alpha,
        a.beta,
        a.gamma,
        a.max_depth,
    )