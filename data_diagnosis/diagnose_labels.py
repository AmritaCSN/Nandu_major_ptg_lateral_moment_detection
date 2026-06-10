"""
diagnose_labels.py
──────────────────
Explain WHY some redteam events don't match auth.txt. For each redteam event,
probe auth.txt with progressively looser criteria and bucket the reason:

  MATCHED_EXACT     : (time,user,src,dst) Success row exists  -> recovered, good
  ONLY_AS_FAILURE   : same key exists but success != Success  -> our Success filter dropped it
  KEY_PRESENT_DIFF  : (time,user,dst) exists but src differs   -> src_comp encoding differs
  TIME_USER_ONLY    : (time,user) exists, hop differs          -> representation differs
  OUTSIDE_WINDOW    : event time is outside our kept window     -> pad/scope too narrow
  ABSENT            : no auth row at that time for that user     -> genuinely not in auth.txt

This reads auth.txt in chunks but ONLY keeps rows whose time is in the set of
redteam timestamps (tiny), so it is cheap regardless of file size.

Usage:
  python diagnose_labels.py --auth /path/auth.txt --redteam /path/redteam.txt
"""

import pandas as pd
import numpy as np
import argparse
from collections import Counter

AUTH_COLS = ["time","src_user","dst_user","src_comp","dst_comp",
             "auth_type","logon_type","orientation","success"]


def main(auth_path, redteam_path, chunksize, pad, max_chunks):
    rt = pd.read_csv(redteam_path, header=None,
                     names=["time","user","src_comp","dst_comp"])
    rt["user"] = rt["user"].str.split("@").str[0]
    rt_times = set(rt["time"].astype(int))
    t_lo, t_hi = int(rt["time"].min()), int(rt["time"].max())
    print(f"[redteam] {len(rt)} events, time range [{t_lo}, {t_hi}] "
          f"(span {t_hi - t_lo} s ≈ {(t_hi-t_lo)/86400:.1f} days)")
    print(f"[note] our labeling window with pad={pad}: "
          f"[{t_lo-pad}, {t_hi+pad}]")

    # collect ALL auth rows whose time is exactly a redteam timestamp
    relevant = []
    reader = pd.read_csv(auth_path, header=None, names=AUTH_COLS,
                         chunksize=chunksize, dtype=str, na_filter=False)
    for ci, chunk in enumerate(reader):
        if max_chunks and ci >= max_chunks:
            print(f"[diag] stopped at {max_chunks} chunks")
            break
        chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
        chunk = chunk.dropna(subset=["time"])
        chunk["time"] = chunk["time"].astype(np.int64)
        cmin = int(chunk["time"].min())
        if cmin > t_hi:
            print(f"[diag] chunk {ci} past last redteam time; stopping")
            break
        hit = chunk[chunk["time"].isin(rt_times)]
        if len(hit):
            hit = hit.copy()
            hit["src_user"] = hit["src_user"].str.split("@").str[0]
            hit["dst_user"] = hit["dst_user"].str.split("@").str[0]
            relevant.append(hit)
        if ci % 50 == 0:
            print(f"  scanned chunk {ci}, collected {sum(len(h) for h in relevant)} "
                  f"time-relevant auth rows")
    A = (pd.concat(relevant, ignore_index=True) if relevant
         else pd.DataFrame(columns=AUTH_COLS))
    print(f"[diag] {len(A)} auth rows fall on redteam timestamps")

    # build lookup sets — check BOTH src_user and dst_user (LANL convention)
    exact_succ = set()        # (time, user, src, dst) with user in src OR dst, Success
    exact_any  = set()        # same, any success value
    tud        = set()        # time, user(any field), dst
    tu         = set()        # time, user(any field)
    tonly      = set()
    for r in A.itertuples(index=False):
        for u in {r.src_user, r.dst_user}:
            k4 = (r.time, u, r.src_comp, r.dst_comp)
            exact_any.add(k4)
            if r.success == "Success":
                exact_succ.add(k4)
            tud.add((r.time, u, r.dst_comp))
            tu.add((r.time, u))
        tonly.add(r.time)

    buckets = Counter()
    examples = {}
    for r in rt.itertuples(index=False):
        k4 = (r.time, r.user, r.src_comp, r.dst_comp)
        if r.time < t_lo - pad or r.time > t_hi + pad:
            b = "OUTSIDE_WINDOW"
        elif k4 in exact_succ:
            b = "MATCHED_EXACT"
        elif k4 in exact_any:
            b = "ONLY_AS_FAILURE"
        elif (r.time, r.user, r.dst_comp) in tud:
            b = "KEY_PRESENT_DIFF_SRC"
        elif (r.time, r.user) in tu:
            b = "TIME_USER_ONLY"
        elif r.time in tonly:
            b = "TIME_ONLY_DIFF_USER"
        else:
            b = "ABSENT"
        buckets[b] += 1
        examples.setdefault(b, (r.time, r.user, r.src_comp, r.dst_comp))

    print("\n" + "=" * 56)
    print("  WHY REDTEAM EVENTS DO / DON'T MATCH")
    print("=" * 56)
    for b in ["MATCHED_EXACT","ONLY_AS_FAILURE","KEY_PRESENT_DIFF_SRC",
              "TIME_USER_ONLY","TIME_ONLY_DIFF_USER","OUTSIDE_WINDOW","ABSENT"]:
        if buckets.get(b):
            print(f"  {b:<22} {buckets[b]:>4}   e.g. {examples[b]}")
    print("=" * 56)
    rec = buckets["MATCHED_EXACT"]
    print(f"  recoverable now (exact success): {rec}/{len(rt)} "
          f"({100*rec/len(rt):.1f}%)")
    gain_fail = buckets.get("ONLY_AS_FAILURE",0)
    gain_src  = buckets.get("KEY_PRESENT_DIFF_SRC",0)
    if gain_fail:
        print(f"  +{gain_fail} more if we DON'T filter to Success only")
    if gain_src:
        print(f"  +{gain_src} more if we match on (time,user,dst) ignoring src_comp")
    if buckets.get("OUTSIDE_WINDOW"):
        print(f"  !! {buckets['OUTSIDE_WINDOW']} are OUTSIDE your window — widen "
              f"--pad or remove --max-chunks; this is the main fixable loss")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", required=True)
    ap.add_argument("--redteam", required=True)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--pad", type=int, default=86400)
    ap.add_argument("--max-chunks", type=int, default=None)
    a = ap.parse_args()
    main(a.auth, a.redteam, a.chunksize, a.pad, a.max_chunks)
