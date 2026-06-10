"""
ablation.py
-----------
Two paper-required additions flagged in review:

(A) ALERT weight ablation — sweep (alpha, beta, gamma) and report PR-AUC + DR,
so the weight choice is empirically justified, not asserted.

(B) External baseline — Isolation Forest (and optionally One-Class SVM) on the
same per-event features, so the PTG is compared against a standard
unsupervised anomaly detector, not only against its own event-level ablation.

Run after producing outputs/data/labeled.parquet.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(__file__))
CORE = os.path.join(ROOT, "core")

sys.path.insert(0, CORE)

from ptg_graph import build_snapshots
from ptg_scorer import Baseline, PTGScorer


def load(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(os.path.splitext(path)[0] + ".csv")


def dr_at(df, col, k):
    ranked = df.sort_values(col, ascending=False).head(k)
    total = int(df["label"].sum())
    return round(int(ranked["label"].sum()) / max(total, 1), 4)


def score_ptg_with_weights(df, baseline, snapshots, alpha, beta, gamma, max_depth):
    scorer = PTGScorer(baseline, alpha, beta, gamma)
    hop_scores = {}

    for ptg in snapshots:
        for scored_path in scorer.score_snapshot(ptg, max_depth=max_depth):
            for event_id in scored_path["event_ids"]:
                if event_id is not None and scored_path["ALERT"] > hop_scores.get(event_id, -1):
                    hop_scores[event_id] = scored_path["ALERT"]

    scores = df.index.map(lambda i: hop_scores.get(i, 0.0)).to_numpy()
    return scores


def weight_ablation(df, max_depth=4, delta=3600, smoothing=1.0):
    df = df.reset_index(drop=True)

    rt_start = int(df[df.label == 1]["time"].min())
    train = df[df["time"] < rt_start]
    if len(train) < 50:
        train = df[df["time"] < df["time"].quantile(0.3)]

    baseline = Baseline(smoothing=smoothing)
    baseline.train(train)

    snapshots = build_snapshots(df, delta_seconds=delta)
    y_true = df["label"].to_numpy()

    grid = [
        (0.5, 0.3, 0.2),
        (0.4, 0.4, 0.2),
        (0.3, 0.5, 0.2),
        (0.34, 0.33, 0.33),
        (0.6, 0.2, 0.2),
        (0.2, 0.6, 0.2),
    ]

    rows = []
    for alpha, beta, gamma in grid:
        scores = score_ptg_with_weights(df, baseline, snapshots, alpha, beta, gamma, max_depth)
        rows.append({
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "PR_AUC": round(float(average_precision_score(y_true, scores)), 5),
            "DR@50": dr_at(df.assign(_s=scores), "_s", 50),
            "DR@100": dr_at(df.assign(_s=scores), "_s", 100),
        })

    return pd.DataFrame(rows)


def isolation_forest_baseline(df, contamination="auto", seed=42):
    """
    Standard unsupervised anomaly detector on simple per-event features.
    Trained unsupervised on benign-context events; scored on all events.
    """
    df = df.reset_index(drop=True)

    feats = pd.DataFrame(index=df.index)
    for col in ["src_user", "src_comp", "dst_comp", "edge_type"]:
        freq = df[col].map(df[col].value_counts(normalize=True))
        feats[col + "_freq"] = freq

    feats["cross_host"] = (df["src_comp"] != df["dst_comp"]).astype(int)
    feats["is_lateral"] = (df["edge_type"] == "lateral").astype(int)

    X = StandardScaler().fit_transform(feats.fillna(0.0))

    rt_start = int(df[df.label == 1]["time"].min())
    train_mask = (df["time"] < rt_start).to_numpy()
    if train_mask.sum() < 50:
        train_mask = (df["time"] < df["time"].quantile(0.3)).to_numpy()

    iso = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=200,
    )
    iso.fit(X[train_mask])

    scores = -iso.score_samples(X)
    y_true = df["label"].to_numpy()

    return {
        "PR_AUC": round(float(average_precision_score(y_true, scores)), 5),
        "DR@50": dr_at(df.assign(_s=scores), "_s", 50),
        "DR@100": dr_at(df.assign(_s=scores), "_s", 100),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/labeled.parquet")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--delta", type=int, default=3600)
    a = ap.parse_args()

    df = load(a.data)

    print("\n=== (A) ALERT weight ablation ===")
    wa = weight_ablation(df, max_depth=a.max_depth, delta=a.delta)
    print(wa.to_string(index=False))

    print("\n=== (B) Isolation Forest external baseline ===")
    iso = isolation_forest_baseline(df)
    print(iso)

    os.makedirs("outputs/results", exist_ok=True)
    wa.to_csv("outputs/results/weight_ablation.csv", index=False)
    with open("outputs/results/iso_baseline.json", "w") as f:
        json.dump(iso, f, indent=2)

    print("\n[saved] outputs/results/weight_ablation.csv, outputs/results/iso_baseline.json")