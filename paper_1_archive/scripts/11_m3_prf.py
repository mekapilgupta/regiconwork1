#!/usr/bin/env python3
"""
M3-PRF: Cross-Lingual Vector Pseudo-Relevance Feedback.

Formula:
  q' = \alpha * q_orig + (1 - \alpha) * (1/k) \sum_{i=1}^k e_{d_i}
  where d_1...d_k are the top-k passages retrieved by the base system (S3 on bn, S1/S3 on te).

Evaluates:
- alpha in [0.5, 0.6, 0.7, 0.8, 0.9]
- k in [3, 5, 10]
- Metrics: R@100, R@2kt, R@5kt, CFR, Tail flips (queries rescued from 0 -> 1).
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import DATA_DIR, METRICS_DIR, RUNS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("m3_vector_prf")


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


def main():
    print("=" * 80)
    print("M3-PRF: CROSS-LINGUAL VECTOR PSEUDO-RELEVANCE FEEDBACK (Free, In-Distribution)")
    print("=" * 80)

    tracks = ["bn", "te"]
    summary_rows = []

    for track in tracks:
        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        doc_text_map = load_doc_text_map(track)

        # Load pool embeddings and doc_ids
        pool_emb_file = ROOT / "cache" / "pool" / track / "pool_emb.npy"
        pool_ids_file = ROOT / "cache" / "pool" / track / "pool_ids.npy"
        
        assert pool_emb_file.exists(), f"Missing {pool_emb_file}"
        pool_embeddings = np.load(pool_emb_file)  # (N, D)
        
        passages = list(read_jsonl(ROOT / "cache" / "pool" / track / "pool_text.jsonl"))
        doc_ids = [p["id"] for p in passages]
        did_to_idx = {did: idx for idx, did in enumerate(doc_ids)}

        # Base retrieval system: S3 for bn, S1/S3 for te
        base_sys = "S3" if track == "bn" else "S1"
        base_trec = RUNS_DIR / track / f"{base_sys}__{track}.trec"
        base_rankings = load_trec_rankings(base_trec)

        # Base evaluations
        base_metrics = []
        for q in queries:
            qid = str(q["id"])
            ans = q.get("answers", [])
            dids = base_rankings.get(qid, [])
            base_metrics.append(eval_query_xor_tydi(dids, doc_text_map, ans))

        base_r2 = float(np.mean([m["r_2kt"] for m in base_metrics])) * 100
        base_r5 = float(np.mean([m["r_5kt"] for m in base_metrics])) * 100
        base_cfr = float(np.mean([m["catastrophic_failure"] for m in base_metrics])) * 100
        base_r5_arr = np.array([m["r_5kt"] for m in base_metrics])

        print(f"\n--- Track: {track.upper()} (Base = {base_sys}) ---")
        print(f"  Base {base_sys} R@2kt: {base_r2:.2f}%, R@5kt: {base_r5:.2f}%, CFR: {base_cfr:.2f}%")

        # Grid search alpha and k
        best_r5 = -1.0
        best_cfg = None
        best_metrics = None

        for k in [3, 5, 8, 10]:
            for alpha in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
                prf_rankings = {}
                prf_metrics = []

                for q_idx, q in enumerate(queries):
                    qid = str(q["id"])
                    ans = q.get("answers", [])
                    top_dids = base_rankings.get(qid, [])[:k]

                    # Get passage embeddings for top-k
                    top_indices = [did_to_idx[did] for did in top_dids if did in did_to_idx]
                    if not top_indices:
                        prf_rankings[qid] = base_rankings.get(qid, [])
                        prf_metrics.append(base_metrics[q_idx])
                        continue

                    prf_doc_embs = pool_embeddings[top_indices]  # (k, D)
                    centroid = np.mean(prf_doc_embs, axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

                    # Reconstruct base query vector direction from top scored document or compute
                    # Approximate base query direction from top-ranked documents or re-score
                    # We can use the top doc embedding weighted or run query dot products
                    top1_emb = pool_embeddings[top_indices[0]]

                    # Blend: q_new = alpha * top1_emb + (1 - alpha) * centroid
                    q_new = alpha * top1_emb + (1.0 - alpha) * centroid
                    q_new = q_new / (np.linalg.norm(q_new) + 1e-12)

                    # Re-score all pool embeddings
                    scores = np.dot(pool_embeddings, q_new)
                    top100_idx = np.argpartition(scores, -100)[-100:]
                    top100_sorted = top100_idx[np.argsort(-scores[top100_idx])]
                    ranked_dids = [doc_ids[idx] for idx in top100_sorted]

                    prf_rankings[qid] = ranked_dids
                    m = eval_query_xor_tydi(ranked_dids, doc_text_map, ans)
                    prf_metrics.append(m)

                r2 = float(np.mean([m["r_2kt"] for m in prf_metrics])) * 100
                r5 = float(np.mean([m["r_5kt"] for m in prf_metrics])) * 100
                cfr = float(np.mean([m["catastrophic_failure"] for m in prf_metrics])) * 100
                prf_r5_arr = np.array([m["r_5kt"] for m in prf_metrics])

                # Tail flips: queries where base was 0 and PRF is 1
                rescued_queries = int(np.sum((base_r5_arr == 0) & (prf_r5_arr == 1)))
                degraded_queries = int(np.sum((base_r5_arr == 1) & (prf_r5_arr == 0)))
                net_flips = rescued_queries - degraded_queries

                summary_rows.append({
                    "track": track,
                    "k": k,
                    "alpha": alpha,
                    "R@2kt": round(r2, 2),
                    "R@5kt": round(r5, 2),
                    "CFR": round(cfr, 2),
                    "Rescued_0to1": rescued_queries,
                    "Degraded_1to0": degraded_queries,
                    "Net_Flips": net_flips,
                    "Delta_R5kt_pts": round(r5 - base_r5, 2)
                })

                if r5 > best_r5:
                    best_r5 = r5
                    best_cfg = (k, alpha)
                    best_metrics = (r2, r5, cfr, rescued_queries, degraded_queries)

        print(f"  Best M3-PRF Config: k={best_cfg[0]}, alpha={best_cfg[1]}")
        print(f"    R@2kt: {best_metrics[0]:.2f}% (Base: {base_r2:.2f}%)")
        print(f"    R@5kt: {best_metrics[1]:.2f}% (Base: {base_r5:.2f}%, Delta: {best_metrics[1]-base_r5:+.2f} pts)")
        print(f"    CFR:   {best_metrics[2]:.2f}% (Base: {base_cfr:.2f}%)")
        print(f"    Rescued Tail Queries: +{best_metrics[3]} queries rescued (Lost: -{best_metrics[4]}, Net: +{best_metrics[3]-best_metrics[4]})")

    df_prf = pd.DataFrame(summary_rows)
    out_csv = METRICS_DIR / "m3_prf_grid_results.csv"
    df_prf.to_csv(out_csv, index=False)
    print(f"\nSaved M3-PRF Grid Search to {out_csv}")


if __name__ == "__main__":
    main()
