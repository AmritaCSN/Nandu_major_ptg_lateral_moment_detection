"""
ptg_evaluate.py
---------------

End-to-end PTG evaluation on real LANL labels.

labeled events -> train baseline (benign, pre-attack) -> build PTG snapshots
-> score every identity-path (PTS/EPL/IPI/ALERT)
-> map each path's ALERT back onto its constituent hops
-> a (time,user,src,dst) hop is malicious per redteam.txt
-> PR-AUC, detection@budget, recall@FPR, MTTD

Path->hop mapping:
A path is a chain of hops; we attribute the path's ALERT score to each hop (edge) on it.
A hop's final score is the MAX ALERT over all paths that include it. This yields a per-hop
score we can evaluate against per-hop labels — the same ground truth granularity as redteam.txt.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ptg_graph import build_snapshots
from ptg_scorer import Baseline, PTGScorer


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "outputs" / "data" / "labeled.parquet"
RESULTS_DIR = BASE_DIR / "outputs" / "results"


def validate_input_dataframe(df):
    required_columns = {"time", "label"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def load_input_dataframe(data_path):
    data_path = Path(data_path)

    if data_path.exists():
        if data_path.suffix == ".parquet":
            return pd.read_parquet(data_path)
        if data_path.suffix == ".csv":
            return pd.read_csv(data_path)

    parquet_fallback = data_path if data_path.suffix == ".parquet" else data_path.with_suffix(".parquet")
    csv_fallback = data_path if data_path.suffix == ".csv" else data_path.with_suffix(".csv")

    if parquet_fallback.exists():
        return pd.read_parquet(parquet_fallback)
    if csv_fallback.exists():
        return pd.read_csv(csv_fallback)

    raise FileNotFoundError(f"Could not find input data at {data_path}, {parquet_fallback}, or {csv_fallback}")


def evaluate(
    df,
    delta=3600,
    stride=None,
    max_depth=4,
    smoothing=1.0,
    alpha=0.4,
    beta=0.4,
    gamma=0.2,
    verbose=True,
):
    validate_input_dataframe(df)

    malicious_rows = df[df["label"] == 1]
    if malicious_rows.empty:
        raise ValueError("Input dataframe contains no positive labels; evaluation requires at least one malicious event.")

    rt_start = int(malicious_rows["time"].min())
    train = df[df["time"] < rt_start].copy()

    if len(train) < 50:
        cut = df["time"].quantile(0.3)
        train = df[df["time"] < cut].copy()

    baseline = Baseline(smoothing=smoothing)
    baseline.train(train)
    scorer = PTGScorer(baseline, alpha, beta, gamma)

    if verbose:
        print(
            f"[ptg-eval] baseline trained on {len(train):,} benign events, "
            f"{len(baseline.p_edge)} identities"
        )

    df = df.reset_index(drop=True)
    snapshots = build_snapshots(df, delta_seconds=delta, stride_seconds=stride)

    if verbose:
        print(
            f"[ptg-eval] built {len(snapshots)} PTG snapshots "
            f"(delta={delta}s, stride={stride or delta / 2}s)"
        )

    hop_score = {}
    n_paths = 0

    for ptg in snapshots:
        scored_paths = scorer.score_snapshot(ptg, max_depth=max_depth)
        n_paths += len(scored_paths)

        for scored_path in scored_paths:
            for event_id in scored_path["event_ids"]:
                if event_id is None:
                    continue
                if scored_path["ALERT"] > hop_score.get(event_id, -1):
                    hop_score[event_id] = scored_path["ALERT"]

    if verbose:
        print(
            f"[ptg-eval] scored {n_paths:,} identity-paths, "
            f"{len(hop_score):,} distinct events flagged"
        )

    scored_df = df.copy()
    scored_df["score_ptg"] = scored_df.index.map(lambda idx: hop_score.get(idx, 0.0))

    y_true = scored_df["label"].to_numpy()
    scores = scored_df["score_ptg"].to_numpy()

    results = {
        "PR_AUC": round(float(average_precision_score(y_true, scores)), 5),
        "ROC_AUC": round(float(roc_auc_score(y_true, scores)), 5),
        "n_events": int(len(scored_df)),
        "n_malicious": int(y_true.sum()),
        "n_paths_scored": int(n_paths),
        "det_at_budget": {},
        "recall_at_fpr": {},
    }

    ranked = scored_df.sort_values("score_ptg", ascending=False).reset_index(drop=True)
    total_malicious = int(y_true.sum())

    for k in [10, 25, 50, 100, 200]:
        caught = int(ranked.head(k)["label"].sum())
        results["det_at_budget"][k] = {
            "caught": caught,
            "total": total_malicious,
            "detection_rate": round(caught / max(total_malicious, 1), 4),
            "precision": round(caught / k, 4),
        }

    ranked_labels = ranked["label"].to_numpy()
    n_neg = int((ranked_labels == 0).sum())
    n_pos = int((ranked_labels == 1).sum())
    cum_fp = np.cumsum(ranked_labels == 0)
    cum_tp = np.cumsum(ranked_labels == 1)

    for fpr in [0.0001, 0.001, 0.01]:
        eligible = np.where(cum_fp <= fpr * n_neg)[0]
        recall = (cum_tp[eligible[-1]] / max(n_pos, 1)) if len(eligible) else 0.0
        results["recall_at_fpr"][fpr] = round(float(recall), 4)

    malicious_only = scored_df[scored_df["label"] == 1]
    threshold = scored_df["score_ptg"].quantile(0.999)
    true_positive_alerts = scored_df[(scored_df["score_ptg"] >= threshold) & (scored_df["label"] == 1)]

    if not true_positive_alerts.empty:
        span = max(malicious_only["time"].max() - malicious_only["time"].min(), 1)
        results["mttd"] = {
            "first_tp_time": int(true_positive_alerts["time"].min()),
            "frac_into_window": round(
                (true_positive_alerts["time"].min() - malicious_only["time"].min()) / span,
                4,
            ),
            "n_tp_alerts": int(len(true_positive_alerts)),
        }
    else:
        results["mttd"] = {
            "first_tp_time": None,
            "frac_into_window": None,
            "n_tp_alerts": 0,
        }

    return results, scored_df


def print_report(results):
    print("\n" + "=" * 56)
    print(" PTG DETECTION - LANL red-team ground truth")
    print("=" * 56)
    print(
        f" events={results['n_events']:,} malicious={results['n_malicious']} "
        f"paths_scored={results['n_paths_scored']:,}"
    )
    print(f" PR-AUC = {results['PR_AUC']} ROC-AUC = {results['ROC_AUC']}")

    print("\n Detection rate @ alert budget:")
    for k, values in results["det_at_budget"].items():
        print(
            f" top-{k:<4} caught {values['caught']:>3}/{values['total']:<3} "
            f"DR={values['detection_rate']:.3f} precision={values['precision']:.3f}"
        )

    print("\n Recall @ fixed FPR:")
    for fpr, recall in results["recall_at_fpr"].items():
        print(f" FPR={fpr:<8} recall={recall:.3f}")

    print(f"\n MTTD: {results['mttd']}")
    print("=" * 56)


def save_outputs(scored_df, results, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    scored_path = results_dir / "ptg_scored.csv"
    results_path = results_dir / "ptg_results.json"

    scored_df.to_csv(scored_path, index=False)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[saved] {scored_path}, {results_path}")
    return {
        "ptg_scored_csv": str(scored_path),
        "ptg_results_json": str(results_path),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--delta", type=int, default=3600)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--smoothing", type=float, default=1.0)
    return parser


def main(args=None):
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)

    df = load_input_dataframe(parsed.data)
    results, scored_df = evaluate(
        df,
        delta=parsed.delta,
        stride=parsed.stride,
        max_depth=parsed.max_depth,
        smoothing=parsed.smoothing,
    )

    print_report(results)
    saved_paths = save_outputs(scored_df, results)

    return {
        "success": True,
        "results": results,
        "saved_paths": saved_paths,
    }


if __name__ == "__main__":
    main()