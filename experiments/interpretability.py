"""
interpretability.py (v2 — campaign-level reconstruction)
─────────────────────────────────────────────────────────
Quantify PTG's interpretability advantage HONESTLY.

Previous version reported each scored path separately, so an identity that
touched 19 hosts produced 19 length-1 "chains" — making chain-length look like 2
and a localization metric trivially 1.0. This version reconstructs CAMPAIGNS:
for each identity, aggregate all PTG-flagged hops into one campaign object
(hosts reached, ordered hops, edge mix, IPI, max alert, red-team overlap), rank
campaigns by max alert, and measure honest metrics on them.
"""

import pandas as pd
import numpy as np
import argparse, os, sys, json
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(__file__))
CORE = os.path.join(ROOT, "core")

if CORE not in sys.path:
    sys.path.insert(0, CORE)

from ptg_graph import build_snapshots
from ptg_scorer import Baseline, PTGScorer

LATERAL_EDGES = {"lateral", "remote_login"}

def load(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(os.path.splitext(path)[0] + ".csv")

def reconstruct_campaigns(df, delta=3600, max_depth=4, smoothing=1.0,
                          alpha=0.3, beta=0.5, gamma=0.2):
    df = df.reset_index(drop=True)
    rt_start = int(df[df.label == 1]["time"].min())
    train = df[df["time"] < rt_start]
    if len(train) < 50:
        train = df[df["time"] < df["time"].quantile(0.3)]
    bl = Baseline(smoothing=smoothing); bl.train(train)
    scorer = PTGScorer(bl, alpha, beta, gamma)
    snaps = build_snapshots(df, delta_seconds=delta)

    mal_idx = set(df.index[df.label == 1].tolist())
    src_map = df["src_comp"].to_dict()
    dst_map = df["dst_comp"].to_dict()
    edge_map = df["edge_type"].to_dict()
    time_map = df["time"].to_dict()

    agg = defaultdict(lambda: {"event_ids": set(), "max_alert": 0.0,
                               "max_pts": 0.0, "max_epl": 0.0, "max_ipi": 0})
    for ptg in snaps:
        for sp in scorer.score_snapshot(ptg, max_depth=max_depth):
            a = agg[sp["identity"]]
            for eid in sp["event_ids"]:
                if eid is not None:
                    a["event_ids"].add(eid)
            a["max_alert"] = max(a["max_alert"], sp["ALERT"])
            a["max_pts"] = max(a["max_pts"], sp["PTS"])
            a["max_epl"] = max(a["max_epl"], sp["EPL"])
            a["max_ipi"] = max(a["max_ipi"], sp["IPI"])

    campaigns = []
    for ident, a in agg.items():
        eids = sorted(a["event_ids"], key=lambda e: time_map.get(e, 0))
        if not eids:
            continue
        hosts = set(); hops = []; edge_mix = defaultdict(int); n_mal = 0
        for e in eids:
            s, d = src_map.get(e), dst_map.get(e)
            hosts.add(s); hosts.add(d); hops.append((s, d))
            edge_mix[edge_map.get(e)] += 1
            if e in mal_idx:
                n_mal += 1
        campaigns.append({
            "identity": ident, "max_alert": round(a["max_alert"], 4),
            "PTS": round(a["max_pts"], 4), "EPL": round(a["max_epl"], 4),
            "IPI": a["max_ipi"], "n_hops": len(hops),
            "n_distinct_hosts": len(hosts), "edge_mix": dict(edge_mix),
            "n_redteam_hops": n_mal, "is_true": n_mal > 0,
            "hosts": sorted(hosts),
            "time_span": [int(time_map.get(eids[0], 0)),
                          int(time_map.get(eids[-1], 0))],
        })
    campaigns.sort(key=lambda c: -c["max_alert"])
    return campaigns, df

def evaluate(df, K=25, **kw):
    campaigns, df = reconstruct_campaigns(df, **kw)
    top = campaigns[:K]
    tp = [c for c in top if c["is_true"]]
    purity = len(tp) / max(len(top), 1)
    rt_hosts = set(df[df.label == 1]["dst_comp"]) | set(df[df.label == 1]["src_comp"])
    surfaced = set()
    for c in tp:
        surfaced |= set(c["hosts"])
    footprint_recall = len(surfaced & rt_hosts) / max(len(rt_hosts), 1)
    hosts_per = np.mean([c["n_distinct_hosts"] for c in tp]) if tp else 0.0
    hops_per = np.mean([c["n_hops"] for c in tp]) if tp else 0.0
    rich = [1 + c["n_distinct_hosts"] + len(c["edge_mix"]) + 4 for c in top]
    ptg_rich = float(np.mean(rich)) if rich else 0.0
    result = {
        "top_K_campaigns": K,
        "true_campaigns_in_topK": len(tp),
        "campaign_purity": round(purity, 4),
        "host_footprint_recall": round(footprint_recall, 4),
        "mean_distinct_hosts_per_true_campaign": round(float(hosts_per), 2),
        "mean_hops_per_true_campaign": round(float(hops_per), 2),
        "mean_structured_fields_per_campaign_PTG": round(ptg_rich, 2),
        "structured_fields_per_alert_baseline": 1.0,
        "context_richness_ratio": round(ptg_rich / 1.0, 1),
    }
    return result, campaigns

def print_report(res, campaigns, k=3):
    print("\n" + "=" * 60)
    print(" INTERPRETABILITY — campaign-level reconstruction")
    print("=" * 60)
    for kk, v in res.items():
        print(f" {kk:<46} {v}")
    print("\n Example reconstructed campaigns (one alert per identity):")
    for c in [c for c in campaigns if c["is_true"]][:k]:
        print(f"\n [CAMPAIGN alert={c['max_alert']}] identity={c['identity']}")
        print(f" reached {c['n_distinct_hosts']} hosts over {c['n_hops']} hops"
              f" (IPI={c['IPI']})")
        print(f" edge mix: {c['edge_mix']}")
        print(f" red-team hops: {c['n_redteam_hops']}/{c['n_hops']}")
        print(f" hosts: {c['hosts'][:8]}{' ...' if len(c['hosts'])>8 else ''}")
        print(f" PTS={c['PTS']} EPL={c['EPL']}")
    print("\n A feature-vector baseline emits one scalar per event and cannot")
    print(" group these into a per-identity campaign at all.")
    print("=" * 60)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/labeled_small.parquet")
    ap.add_argument("--delta", type=int, default=3600)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--K", type=int, default=25)
    a = ap.parse_args()
    df = load(a.data)
    res, campaigns = evaluate(df, K=a.K, delta=a.delta, max_depth=a.max_depth)
    print_report(res, campaigns, k=3)
    os.makedirs("out", exist_ok=True)
    with open("out/interpretability.json", "w") as f:
        json.dump(res, f, indent=2)
    with open("out/example_campaigns.json", "w") as f:
        json.dump([c for c in campaigns if c["is_true"]][:10], f, indent=2, default=str)
    print("\n[saved] out/interpretability.json, out/example_campaigns.json")