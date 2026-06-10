"""
baselines_fair.py
-----------------
FAIR external baselines for the PTG comparison.

Motivation: the original Isolation Forest scored each event with only a few raw
features and no sequence/identity/window context — it was handicapped relative
to PTG (which scores PATHS over an identity's behavior in a window). A baseline
that scores below random (PR-AUC ~ prevalence floor or worse) is usually the
sign of a feature set with too little signal for the task, not of an impossible
task.

Fix: engineer per-event features that summarize the SAME identity-behavioral
information PTG derives structurally — distinct hosts touched in window, lateral
counts, transition-type counts, events-per-window, etc. Then compare, on
equivalent information:
- Isolation Forest (unsupervised)
- One-Class SVM (unsupervised, trained on benign)
- Logistic Regression (supervised, temporal split — reported separately)

The honest claim becomes: "given equivalent identity-behavioral information,
scoring privilege-transition PATHS (PTG) outperforms flattening that information
into feature-vector anomaly detection — and uniquely yields an interpretable
attack chain."
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


EDGE_TYPES = ["login", "remote_login", "lateral"]


def load(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(os.path.splitext(path)[0] + ".csv")


def dr_at(df, score, k):
    ranked = pd.DataFrame(
        {"label": df["label"].to_numpy(), "s": score}
    ).sort_values("s", ascending=False).head(k)
    total = int(df["label"].sum())
    return round(int(ranked["label"].sum()) / max(total, 1), 4)


def precision_at(df, score, k):
    ranked = pd.DataFrame(
        {"label": df["label"].to_numpy(), "s": score}
    ).sort_values("s", ascending=False).head(k)
    return round(int(ranked["label"].sum()) / k, 4)


def build_fair_features(df, delta=3600):
    """
    Per-event features that summarize the identity's behavior in the sliding
    window ending at that event — the SAME information PTG sees structurally,
    flattened into a vector. No graph, no path: that is the point of contrast.
    """
    df = df.sort_values(["src_user", "time"]).reset_index(drop=True)
    feats = []

    for ident, g in df.groupby("src_user", sort=False):
        times = g["time"].to_numpy()
        dsts = g["dst_comp"].to_numpy()
        srcs = g["src_comp"].to_numpy()
        edges = g["edge_type"].to_numpy()
        n = len(g)
        idx = g.index.to_numpy()

        for i in range(n):
            t = times[i]
            lo = np.searchsorted(times, t - delta, side="left")

            win_edges = edges[lo:i + 1]
            win_dsts = dsts[lo:i + 1]

            row = {
                "idx": idx[i],
                "win_events": len(win_edges),
                "win_distinct_hosts": len(set(win_dsts)),
                "win_lateral": int(np.sum(win_edges == "lateral")),
                "win_remote": int(np.sum(win_edges == "remote_login")),
                "win_login": int(np.sum(win_edges == "login")),
                "is_lateral": int(edges[i] == "lateral"),
                "is_cross_host": int(dsts[i] != srcs[i]),
                "id_total_events": n,
                "id_lateral_rate": float(np.mean(edges == "lateral")),
            }
            feats.append(row)

    fdf = pd.DataFrame(feats).set_index("idx").sort_index()
    df_sorted = df.sort_index()
    fdf = fdf.reindex(df_sorted.index)
    return df_sorted, fdf


def run_baselines(df, delta=3600, seed=42):
    df2, fdf = build_fair_features(df, delta=delta)
    y = df2["label"].to_numpy()

    feat_cols = list(fdf.columns)
    X = StandardScaler().fit_transform(fdf[feat_cols].fillna(0.0).to_numpy())

    rt_start = int(df2[df2.label == 1]["time"].min())
    train_mask = (df2["time"] < rt_start).to_numpy()
    if train_mask.sum() < 50:
        train_mask = (df2["time"] < df2["time"].quantile(0.3)).to_numpy()

    prevalence = float(y.mean())
    results = {"prevalence_PR_AUC_floor": round(prevalence, 6)}

    iso = IsolationForest(
        contamination="auto",
        random_state=seed,
        n_estimators=200,
    )
    iso.fit(X[train_mask])
    s_iso = -iso.score_samples(X)
    results["IsolationForest"] = {
        "PR_AUC": round(float(average_precision_score(y, s_iso)), 5),
        "ROC_AUC": round(float(roc_auc_score(y, s_iso)), 5),
        "DR@50": dr_at(df2, s_iso, 50),
        "DR@100": dr_at(df2, s_iso, 100),
        "P@50": precision_at(df2, s_iso, 50),
        "P@100": precision_at(df2, s_iso, 100),
    }

    benign_idx = np.where(train_mask)[0]
    if len(benign_idx) > 20000:
        rng = np.random.default_rng(seed)
        benign_idx = rng.choice(benign_idx, 20000, replace=False)

    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(X[benign_idx])
    s_svm = -ocsvm.score_samples(X)
    results["OneClassSVM"] = {
        "PR_AUC": round(float(average_precision_score(y, s_svm)), 5),
        "ROC_AUC": round(float(roc_auc_score(y, s_svm)), 5),
        "DR@50": dr_at(df2, s_svm, 50),
        "DR@100": dr_at(df2, s_svm, 100),
        "P@50": precision_at(df2, s_svm, 50),
        "P@100": precision_at(df2, s_svm, 100),
    }

    order = np.argsort(df2["time"].to_numpy())
    tr = order[: len(order) // 2]
    te = order[len(order) // 2 :]

    if y[tr].sum() >= 5 and y[te].sum() >= 5:
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X[tr], y[tr])
        s_lr_te = lr.predict_proba(X[te])[:, 1]
        results["LogisticRegression_temporal"] = {
            "PR_AUC": round(float(average_precision_score(y[te], s_lr_te)), 5),
            "ROC_AUC": round(float(roc_auc_score(y[te], s_lr_te)), 5),
            "note": "supervised, temporal 50/50 split, test half only",
        }
    else:
        results["LogisticRegression_temporal"] = {
            "note": "insufficient positives in one split half for supervised LR"
        }

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/labeled_small.parquet")
    ap.add_argument("--delta", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = load(a.data)

    print(f"[baselines] {len(df):,} events, {int(df.label.sum())} malicious "
          f"(prevalence {df.label.mean():.6f})")

    res = run_baselines(df, delta=a.delta, seed=a.seed)
    print(json.dumps(res, indent=2))

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/baselines_fair.json", "w") as f:
        json.dump(res, f, indent=2)

    print("\n[saved] outputs/results/baselines_fair.json")