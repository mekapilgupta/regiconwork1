#!/usr/bin/env python3
"""
Stage 4: GPU Offline Retrieval & Evaluation Stage (GPU, OFFLINE).

- Absolutely NO network access (assert_offline).
- Loads pinned encoder (BAAI/bge-m3 or intfloat/multilingual-e5-small) offline.
- Builds DenseIndex and BM25Index over the candidate passage pools.
- Runs Systems S1–S9 and Ablations A1–A4.
- Writes TREC run files to runs/<system>__<track>.trec.
- Computes official metrics:
  * bn/te: R@2kt, R@5kt, catastrophic failure rate
  * hi: MRR@10, Recall@20, Recall@100, catastrophic failure rate
  * hi_mono: nDCG@10, Recall@100, MRR@10
- Computes 95% Bootstrap Confidence Intervals and Paired Significance Tests.
- Saves results to metrics/.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    METRICS_DIR,
    RUNS_DIR,
    assert_offline,
    get_logger,
)
from src.embeddings import DenseEncoder
from src.metrics import (
    bootstrap_ci,
    eval_query_mmarco,
    eval_query_ndcg_10,
    eval_query_xor_tydi,
    paired_bootstrap_test,
)
from src.retrieval import BM25Index, DenseIndex
from src.systems import ALL_SYSTEMS
from src.utils_io import mark_done, read_jsonl

log = get_logger("04_gpu_stage")


def build_candidate_pool(
    records: list[dict[str, Any]],
    track: str,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Builds an authentic evaluation passage pool from data/corpus/wiki_passages.jsonl."""
    doc_ids = []
    documents = []
    doc_text_map = {}
    seen_ids = set()

    corpus_file = DATA_DIR / "corpus" / "wiki_passages.jsonl"
    if corpus_file.exists():
        for p in read_jsonl(corpus_file):
            pid = str(p["id"])
            ptext = p.get("text", "").strip()
            if pid not in seen_ids and ptext:
                seen_ids.add(pid)
                doc_ids.append(pid)
                documents.append(ptext)
                doc_text_map[pid] = ptext

    # If records explicitly contain candidate passages
    for r in records:
        cand_passages = r.get("candidate_passages", []) + r.get("relevant_passages", [])
        for p in cand_passages:
            pid = str(p.get("id", ""))
            ptext = p.get("text", "").strip()
            if pid and ptext and pid not in seen_ids:
                seen_ids.add(pid)
                doc_ids.append(pid)
                documents.append(ptext)
                doc_text_map[pid] = ptext

    # Hard Validation Guards per spec:
    assert len(doc_ids) > 0, f"Candidate pool is empty for track '{track}'!"
    
    # 1. Assert doc IDs are authentic corpus passage IDs, not query IDs
    for pid in doc_ids[:50]:
        assert not pid.startswith("doc_-") and not pid.startswith("doc_10"), f"Contaminated document ID found: {pid}"
        
    # 2. Assert no document is identical to any query gloss (0% query leakage)
    query_glosses = {
        (r.get("llm_translation") or r.get("eng_translation") or "").strip().lower()
        for r in records
        if (r.get("llm_translation") or r.get("eng_translation"))
    }
    for doc in documents:
        assert doc.strip().lower() not in query_glosses, "CRITICAL ERROR: Document text matches query gloss! Corpus contaminated."
        
    # 3. Assert passages are genuine prose (not short questions ending in '?')
    question_like_count = sum(1 for d in documents if d.strip().endswith("?") and len(d.split()) < 15)
    assert question_like_count == 0, f"Found {question_like_count} short question strings in corpus instead of Wikipedia prose!"

    log.info("Validated authentic candidate pool for track '%s': %d passages.", track, len(doc_ids))
    return doc_ids, documents, doc_text_map


def write_trec_run(
    run_path: Path,
    system_name: str,
    query_results: dict[str, list[tuple[str, float]]],
):
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, ranked_list in query_results.items():
            for rank, (doc_id, score) in enumerate(ranked_list, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {system_name}\n")


def evaluate_track(
    track: str,
    records: list[dict[str, Any]],
    encoder: DenseEncoder,
    device: str = "cpu",
):
    log.info("=== Running Offline Retrieval for Track '%s' (%d queries) ===", track, len(records))

    doc_ids, documents, doc_text_map = build_candidate_pool(records, track)

    # 1. Compute and log Pool Recall
    if track in ("bn", "te"):
        queries_with_gold = 0
        for r in records:
            ans_list = [a.strip().lower() for a in r.get("answers", []) if a.strip()]
            if not ans_list:
                continue
            has_ans = any(any(ans in doc.lower() for ans in ans_list) for doc in documents)
            if has_ans:
                queries_with_gold += 1
        pool_recall = (queries_with_gold / len(records)) if records else 0.0
        log.info("============================================================")
        log.info("POOL RECALL for Track '%s': %.2f%% (%d/%d queries have answer in pool)", track, pool_recall * 100, queries_with_gold, len(records))
        log.info("============================================================")
    else:
        queries_with_gold = 0
        for r in records:
            rel_pids = set(str(pid) for pid in r.get("relevant_passage_ids", []))
            if rel_pids and any(pid in rel_pids for pid in doc_ids):
                queries_with_gold += 1
        pool_recall = (queries_with_gold / len(records)) if records else 0.0
        log.info("============================================================")
        log.info("POOL RECALL for Track '%s': %.2f%% (%d/%d queries have relevant doc in pool)", track, pool_recall * 100, queries_with_gold, len(records))
        log.info("============================================================")

    # 2. Build BM25 index
    bm25_index = BM25Index(doc_ids, documents)

    # 3. Build Dense index
    log.info("Encoding passage corpus (%d docs)...", len(documents))
    doc_embeddings = encoder.encode(documents, batch_size=128, is_query=False, normalize=True)
    dense_index = DenseIndex(doc_ids, doc_embeddings, device=device)

    # 4. Run all systems S1-S9, S6_src, and Ablations
    system_runs: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(dict)
    per_query_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    for sys_name, sys_fn in ALL_SYSTEMS.items():
        log.info("Executing system %s on track %s...", sys_name, track)
        for r in tqdm(records, desc=f"{track}:{sys_name}"):
            qid = str(r["id"])

            # Call system with appropriate signature
            if sys_name in ("S1", "S3", "S5", "S8", "S9"):
                ranked = sys_fn(r, dense_index, encoder, top_k=100)
            elif sys_name == "S2":
                ranked = sys_fn(r, bm25_index, top_k=100)
            else:  # S4, S6, S6_src, S7, Ablations
                ranked = sys_fn(r, dense_index, bm25_index, encoder, top_k=100)

            system_runs[sys_name][qid] = ranked
            retrieved_pids = [doc_id for doc_id, _ in ranked]

            # Compute metric per query
            if track in ("bn", "te"):
                answers = r.get("answers", [])
                m = eval_query_xor_tydi(retrieved_pids, doc_text_map, answers)
            elif track == "hi":
                rel_pids = r.get("relevant_passage_ids", [f"doc_{qid}"])
                m = eval_query_mmarco(retrieved_pids, rel_pids)
            else:  # hi_mono
                rel_pids = r.get("relevant_passage_ids", [f"doc_{qid}"])
                m = eval_query_ndcg_10(retrieved_pids, rel_pids)
                m.update(eval_query_mmarco(retrieved_pids, rel_pids))

            per_query_metrics[sys_name][qid] = m

        # Write TREC run
        run_file = RUNS_DIR / f"{sys_name}__{track}.trec"
        write_trec_run(run_file, sys_name, system_runs[sys_name])

    # 5. Aggregate metrics + Bootstrap CI
    log.info("Aggregating metrics and computing Bootstrap CIs...")
    summary_rows = []

    # Choose primary metric for track
    primary_metric_key = "r_5kt" if track in ("bn", "te") else ("mrr_10" if track == "hi" else "ndcg_10")
    s6_scores = np.array([per_query_metrics["S6"][str(r["id"])].get(primary_metric_key, 0.0) for r in records])

    for sys_name in ALL_SYSTEMS.keys():
        row = {"track": track, "system": sys_name, "pool_recall": round(pool_recall, 4)}
        sys_metric_keys = list(per_query_metrics[sys_name][str(records[0]["id"])].keys())

        for k in sys_metric_keys:
            vals = np.array([per_query_metrics[sys_name][str(r["id"])].get(k, 0.0) for r in records])
            mean_v, ci_low, ci_high = bootstrap_ci(vals)
            row[f"{k}_mean"] = round(mean_v, 4)
            row[f"{k}_ci95"] = f"[{round(ci_low, 4)}, {round(ci_high, 4)}]"

        # Paired significance test against S6
        sys_scores = np.array([per_query_metrics[sys_name][str(r["id"])].get(primary_metric_key, 0.0) for r in records])
        p_val = paired_bootstrap_test(s6_scores, sys_scores)
        row["p_val_vs_S6"] = round(p_val, 4)

        summary_rows.append(row)

    # Save summary CSV
    df = pd.DataFrame(summary_rows)
    metrics_path = METRICS_DIR / f"{track}_main.csv"
    df.to_csv(metrics_path, index=False)
    log.info("Track '%s' metrics summary saved to %s", track, metrics_path)


def main():
    parser = argparse.ArgumentParser(description="GPU Offline Stage for Retrieval and Evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    assert_offline()

    encoder = DenseEncoder(device=args.device)

    tracks = ["bn", "te", "hi", "hi_mono"]
    for track in tracks:
        input_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        if not input_file.exists():
            log.warning("Processed file %s not found. Skipping track '%s'.", input_file, track)
            continue

        records = list(read_jsonl(input_file))
        if not records:
            continue

        evaluate_track(track, records, encoder, device=encoder.device)
        mark_done(ROOT / "state", "stage04", track)

    log.info("Stage 4 completed successfully.")


if __name__ == "__main__":
    main()
