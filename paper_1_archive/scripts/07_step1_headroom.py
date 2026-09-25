#!/usr/bin/env python3
"""
Step 1: Headroom Analysis & Routing Complementarity.

Computes:
1. ORACLE_UNION@5kt over {S1, S3, S5, S9} and over {S1, S3} alone.
2. Pairwise complementarity:
   - % queries where S1 fails (R@5kt=0) and S3 succeeds (R@5kt=1).
   - % queries where S3 fails (R@5kt=0) and S1 succeeds (R@5kt=1).
3. Language-only router (bn -> S3, te -> S1) baseline across all metrics.
4. Decision rule evaluation: Headroom = (ORACLE_UNION - best_single).
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import DATA_DIR, RUNS_DIR, METRICS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("07_step1_headroom")


def load_doc_text_map(track: str) -> dict[str, str]:
    pool_file = ROOT / "cache" / "pool" / track / "pool_text.jsonl"
    passages = list(read_jsonl(pool_file))
    return {p["id"]: p["text"] for p in passages}


def load_trec_rankings(trec_file: Path) -> dict[str, list[str]]:
    rankings = defaultdict(list)
    with open(trec_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, did = parts[0], parts[2]
                rankings[qid].append(did)
    return rankings


def eval_system_per_query(ranked_map: dict[str, list[str]], doc_text_map: dict[str, str], queries: list[dict]) -> dict[str, dict[str, float]]:
    per_q = {}
    for q in queries:
        qid = str(q["id"])
        answers = q.get("answers", [])
        dids = ranked_map.get(qid, [])
        m = eval_query_xor_tydi(dids, doc_text_map, answers)
        per_q[qid] = m
    return per_q


def compute_oracle_union(system_ranked_maps: list[dict[str, list[str]]], doc_text_map: dict[str, str], queries: list[dict], budget_chars: int = 5000) -> dict[str, float]:
    """
    Computes Oracle Union: for each query, pool top candidates from each system,
    and evaluate if at least one retrieved passage contains the answer within budget.
    """
    hits_5kt = []
    hits_2kt = []
    cf_list = []

    for q in queries:
        qid = str(q["id"])
        answers = q.get("answers", [])
        
        # Interleave or union candidates from the systems
        seen_dids = set()
        union_dids = []
        
        # Get maximum rank
        max_rank = max(len(s_map.get(qid, [])) for s_map in system_ranked_maps)
        for r in range(max_rank):
            for s_map in system_ranked_maps:
                dids = s_map.get(qid, [])
                if r < len(dids):
                    did = dids[r]
                    if did not in seen_dids:
                        seen_dids.add(did)
                        union_dids.append(did)
        
        # Check if any candidate in the union is relevant
        # Oracle selection: if ANY document in union contains answer, oracle selects it
        hit_5k = 0.0
        hit_2k = 0.0
        hit_any = 0.0
        
        for did in union_dids:
            text = doc_text_map.get(did, "")
            if _is_answer_contained(text, answers):
                hit_any = 1.0
                if len(text) <= 5000:
                    hit_5k = 1.0
                if len(text) <= 2000:
                    hit_2k = 1.0
                break
                
        hits_5kt.append(hit_5k)
        hits_2kt.append(hit_2k)
        cf_list.append(1.0 - hit_any)

    return {
        "r_5kt": float(np.mean(hits_5kt)),
        "r_2kt": float(np.mean(hits_2kt)),
        "cfr": float(np.mean(cf_list)),
        "hits_5kt_per_q": hits_5kt
    }


def main():
    print("=" * 80)
    print("STEP 1: HEADROOM ANALYSIS & COMPLEMENTARITY AUDIT")
    print("=" * 80)

    tracks = ["bn", "te"]
    summary_rows = []

    lang_router_hits_5kt = []
    best_single_hits_5kt = []
    all_queries_count = 0

    for track in tracks:
        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        all_queries_count += len(queries)
        doc_text_map = load_doc_text_map(track)

        # Load runs
        runs = {}
        for s in ["S1", "S2", "S3", "S3r", "S4", "S5", "S5r", "S6", "S7", "S8", "S9"]:
            runs[s] = load_trec_rankings(RUNS_DIR / track / f"{s}__{track}.trec")

        # Evaluate individual systems
        evals = {s: eval_system_per_query(runs[s], doc_text_map, queries) for s in runs}

        s1_r5 = np.array([evals["S1"][str(q["id"])]["r_5kt"] for q in queries])
        s3_r5 = np.array([evals["S3"][str(q["id"])]["r_5kt"] for q in queries])
        s5_r5 = np.array([evals["S5"][str(q["id"])]["r_5kt"] for q in queries])
        s9_r5 = np.array([evals["S9"][str(q["id"])]["r_5kt"] for q in queries])

        # 1. Oracle Unions
        # Oracle over {S1, S3}
        # A query is a hit if S1 OR S3 hits
        oracle_s1_s3_per_q = np.maximum(s1_r5, s3_r5)
        oracle_s1_s3_r5 = float(np.mean(oracle_s1_s3_per_q))

        # Oracle over {S1, S3, S5, S9}
        oracle_4sys_per_q = np.maximum.reduce([s1_r5, s3_r5, s5_r5, s9_r5])
        oracle_4sys_r5 = float(np.mean(oracle_4sys_per_q))

        # 2. Pairwise Complementarity
        # S1 fails (0) and S3 succeeds (1)
        s1_fail_s3_succ = float(np.mean((s1_r5 == 0) & (s3_r5 == 1))) * 100
        # S3 fails (0) and S1 succeeds (1)
        s3_fail_s1_succ = float(np.mean((s3_r5 == 0) & (s1_r5 == 1))) * 100
        # Both succeed
        both_succ = float(np.mean((s1_r5 == 1) & (s3_r5 == 1))) * 100
        # Both fail
        both_fail = float(np.mean((s1_r5 == 0) & (s3_r5 == 0))) * 100

        # Best single in this track
        best_single_name = "S3" if track == "bn" else "S1"
        best_single_r5 = float(np.mean(s3_r5 if track == "bn" else s1_r5))

        # Language router for this track: bn->S3, te->S1
        lang_router_r5 = best_single_r5
        lang_router_hits_5kt.extend(s3_r5 if track == "bn" else s1_r5)

        headroom_s1_s3 = (oracle_s1_s3_r5 - best_single_r5) * 100
        headroom_4sys = (oracle_4sys_r5 - best_single_r5) * 100

        print(f"\n--- TRACK: {track.upper()} (N = {len(queries)}) ---")
        print(f"  S1 (Zero-Shot) R@5kt:            {np.mean(s1_r5)*100:.2f}%")
        print(f"  S3 (MT Dense)  R@5kt:            {np.mean(s3_r5)*100:.2f}%")
        print(f"  Best Single ({best_single_name}) R@5kt:       {best_single_r5*100:.2f}%")
        print(f"  ORACLE UNION {{S1, S3}} R@5kt:      {oracle_s1_s3_r5*100:.2f}%  (Headroom = +{headroom_s1_s3:.2f} pts)")
        print(f"  ORACLE UNION {{S1,S3,S5,S9}} R@5kt: {oracle_4sys_r5*100:.2f}%  (Headroom = +{headroom_4sys:.2f} pts)")
        print(f"  Complementarity:")
        print(f"    * S1 fails & S3 succeeds:      {s1_fail_s3_succ:.2f}% of queries ({int(round(s1_fail_s3_succ*len(queries)/100))}/{len(queries)})")
        print(f"    * S3 fails & S1 succeeds:      {s3_fail_s1_succ:.2f}% of queries ({int(round(s3_fail_s1_succ*len(queries)/100))}/{len(queries)})")
        print(f"    * Both succeed:                {both_succ:.2f}%")
        print(f"    * Both fail:                   {both_fail:.2f}%")

        summary_rows.append({
            "track": track,
            "N": len(queries),
            "S1_R5kt": round(float(np.mean(s1_r5)), 4),
            "S3_R5kt": round(float(np.mean(s3_r5)), 4),
            "Best_Single": best_single_name,
            "Best_Single_R5kt": round(best_single_r5, 4),
            "Oracle_S1_S3_R5kt": round(oracle_s1_s3_r5, 4),
            "Headroom_S1_S3_pts": round(headroom_s1_s3, 2),
            "Oracle_4Sys_R5kt": round(oracle_4sys_r5, 4),
            "Headroom_4Sys_pts": round(headroom_4sys, 2),
            "S1_fail_S3_succ_pct": round(s1_fail_s3_succ, 2),
            "S3_fail_S1_succ_pct": round(s3_fail_s1_succ, 2),
        })

    # Combined Pooled Stats
    overall_lang_router_r5 = float(np.mean(lang_router_hits_5kt)) * 100
    print("\n" + "=" * 80)
    print("OVERALL POOLED SUMMARY (N = 550)")
    print("=" * 80)
    print(f"Language-Only Router (bn->S3, te->S1) R@5kt: {overall_lang_router_r5:.2f}%")

    df_sum = pd.DataFrame(summary_rows)
    out_csv = METRICS_DIR / "step1_headroom_analysis.csv"
    df_sum.to_csv(out_csv, index=False)
    print(f"\nSaved Step 1 Headroom Table to {out_csv}")
    print(df_sum.to_string(index=False))

    # Decision logic
    avg_headroom = df_sum["Headroom_S1_S3_pts"].mean()
    print("\n=== DECISION GATE ===")
    print(f"Average Headroom (Oracle {{S1, S3}} - Best Single): {avg_headroom:.2f} pts")
    if avg_headroom >= 8.0:
        print("-> Decision: Headroom >= 8.0 pts => PROCEED TO FULL AGER (Routing + Verification).")
    elif avg_headroom < 4.0:
        print("-> Decision: Headroom < 4.0 pts => SKIP ROUTING, KEEP ONLY RERANK + VERIFICATION.")
    else:
        print(f"-> Decision: Headroom is {avg_headroom:.2f} pts (Moderate headroom: Proceed with rerank baselines & evaluate router value vs language-only).")


if __name__ == "__main__":
    main()
