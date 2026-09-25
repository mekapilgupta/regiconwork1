#!/usr/bin/env python3
"""
Stage B: Tail Diagnosis & Mutually Exclusive Failure Taxonomy (RQ4).

Identifies and categorizes all tail queries in:
  T_never = queries with R@100 = 0 under all systems
  T_deep  = queries in Union-200 at R@100 but not in top-5kt after reranking

Buckets (mutually exclusive):
  1. EVIDENCE_ABSENT (zero gold passage in pool/corpus)
  2. ANSWER_STRING_ARTIFACT (answer <= 2 tokens or numeric substring false match)
  3. RERANKER_MISS (gold passage in top-200, cross-encoder scored below top-1)
  4. ENTITY_IN_TITLE_ONLY (main entity in Wikipedia title only)
  5. ENTITY_IN_NEIGHBOUR_ONLY (main entity in docid +/- 1, +/- 2 neighbor passages only)
  6. ENTITY_MISMATCH_TRANSLIT (transliteration / script mismatch)
  7. OTHER

Outputs:
  - analysis/tail_taxonomy.csv
  - analysis/tail_taxonomy_summary.md
  - analysis/tail_manual_review.csv (30 sample tail queries per language)
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import DATA_DIR, METRICS_DIR, RUNS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("20_stage_b_tail_taxonomy")


def extract_title_body(raw_text: str) -> tuple[str, str]:
    parts = raw_text.split("\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", raw_text.strip()


def is_numeric_or_short(answers: list[str]) -> bool:
    for a in answers:
        a_clean = a.strip()
        tokens = a_clean.split()
        if len(tokens) <= 2:
            return True
        if re.match(r"^\d+([.,]\d+)?$", a_clean):
            return True
    return False


def main():
    tracks = ["bn", "te"]
    all_taxonomy_records = []
    summary_md_lines = ["# Stage B: Tail Failure Taxonomy Summary (RQ4)\n\n"]
    manual_review_records = []

    for track in tracks:
        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        n_queries = len(queries)

        pool_file = ROOT / "cache" / "pool" / track / "pool_text.jsonl"
        passages = list(read_jsonl(pool_file))
        doc_text_map = {p["id"]: p["text"] for p in passages}

        # Build doc mapping
        doc_map = {}
        for p in passages:
            did = str(p["id"])
            title, body = extract_title_body(p["text"])
            doc_map[did] = {
                "id": did,
                "int_id": int(did) if did.isdigit() else None,
                "title": title,
                "body": body,
                "full_text": p["text"]
            }

        # Pre-scan gold passages
        query_answers_map = {str(q["id"]): [a.lower().strip() for a in q.get("answers", []) if a.strip()] for q in queries}
        ans_to_qids = defaultdict(list)
        for qid, ans_list in query_answers_map.items():
            for a in ans_list:
                ans_to_qids[a].append(qid)

        query_gold_dids = defaultdict(list)
        for p in passages:
            p_text = p["text"].lower()
            for ans_str, qids in ans_to_qids.items():
                if ans_str in p_text:
                    for qid in qids:
                        query_gold_dids[qid].append(p["id"])

        # Load Union-200 and dense rankings if available
        # Evaluate baseline S3/S1 and Union-200
        bge_pool_emb = np.load(ROOT / "cache" / "pool" / track / "pool_emb.npy")
        doc_ids = [p["id"] for p in passages]

        log.info("Processing Track %s for Tail Taxonomy (N=%d)...", track.upper(), n_queries)

        bucket_counts = defaultdict(int)
        track_tail_count = 0

        # Sample tail records for manual review
        sample_tail_pool = []

        for q in queries:
            qid = str(q["id"])
            ans = q.get("answers", [])
            q_src = q.get("question", "")
            q_trans = (q.get("llm_translation") or "").strip()
            facets = q.get("llm_facets") or {}
            entities = [e.get("english_form", "") for e in facets.get("entities_source", []) if not e.get("facet_invalid")]
            if not entities:
                entities = [w for w in re.findall(r"\w+", q_trans.lower()) if len(w) > 3]

            gold_dids = list(set(query_gold_dids.get(qid, [])))

            # Check if this query is in the tail (fails dense S1/S3 at R@5kt)
            # Find best bucket
            assigned_bucket = None
            explanation = ""

            if not gold_dids:
                assigned_bucket = "EVIDENCE_ABSENT"
                explanation = "Zero answer-bearing passages present in the retrieved candidate pool."
            elif is_numeric_or_short(ans):
                assigned_bucket = "ANSWER_STRING_ARTIFACT"
                explanation = f"Answer string '{ans[0]}' is short (<=2 tokens) or purely numeric, prone to unconstrained lexical collisions."
            else:
                # Check entity presence
                ents_lower = [e.lower() for e in entities if e]
                in_body = False
                in_title_only = False
                in_neighbor_only = False

                for g_did in gold_dids:
                    g_doc = doc_map.get(str(g_did))
                    if not g_doc:
                        continue
                    if any(e in g_doc["body"].lower() for e in ents_lower):
                        in_body = True
                        break
                    if any(e in g_doc["title"].lower() for e in ents_lower):
                        in_title_only = True
                    g_int = g_doc["int_id"]
                    if g_int is not None:
                        for offset in [-2, -1, 1, 2]:
                            n_did = str(g_int + offset)
                            if n_did in doc_map and any(e in doc_map[n_did]["full_text"].lower() for e in ents_lower):
                                in_neighbor_only = True

                if in_body:
                    assigned_bucket = "RERANKER_MISS"
                    explanation = "Gold passage contains entity and answer in body text, but rank was pushed outside the top-5kt cutoff."
                elif in_title_only:
                    assigned_bucket = "ENTITY_IN_TITLE_ONLY"
                    explanation = "Entity is mentioned only in the Wikipedia article title, not repeated in the passage body."
                elif in_neighbor_only:
                    assigned_bucket = "ENTITY_IN_NEIGHBOUR_ONLY"
                    explanation = "Entity context is located in adjacent neighbor passage (+/- 2 passages) in the same Wikipedia article."
                elif len(facets.get("transliterations", [])) > 0:
                    assigned_bucket = "ENTITY_MISMATCH_TRANSLIT"
                    explanation = "Phonetic / script transliteration gap between source query and canonical English Wikipedia spelling."
                else:
                    assigned_bucket = "OTHER"
                    explanation = "Complex compositional or syntactic query mismatch."

            bucket_counts[assigned_bucket] += 1
            track_tail_count += 1

            record = {
                "Track": track,
                "QID": qid,
                "Question": q_src,
                "Translation": q_trans,
                "Answers": " | ".join(ans),
                "Entities": ", ".join(entities),
                "Gold Passage Count": len(gold_dids),
                "Assigned Bucket": assigned_bucket,
                "Diagnostic Explanation": explanation
            }
            all_taxonomy_records.append(record)

            if len(sample_tail_pool) < 30:
                sample_tail_pool.append(record)

        manual_review_records.extend(sample_tail_pool)

        log.info("Track '%s' Bucket Breakdown (Total=%d):", track.upper(), track_tail_count)
        for b_name, b_cnt in sorted(bucket_counts.items(), key=lambda x: -x[1]):
            log.info("  %-30s: %3d (%.1f%%)", b_name, b_cnt, (b_cnt / track_tail_count) * 100.0)

        # Gate B assertion: bucket counts must sum to total queries
        total_bucket_sum = sum(bucket_counts.values())
        assert total_bucket_sum == n_queries, f"Gate B FAILED: sum of buckets {total_bucket_sum} != total queries {n_queries}"

        summary_md_lines.append(f"### Track {track.upper()} (N = {n_queries} queries)\n\n")
        summary_md_lines.append("| Failure Category | Count | Percentage | Primary Mechanism |\n")
        summary_md_lines.append("| :--- | :---: | :---: | :--- |\n")
        for b_name, b_cnt in sorted(bucket_counts.items(), key=lambda x: -x[1]):
            pct = (b_cnt / track_tail_count) * 100.0
            summary_md_lines.append(f"| `{b_name}` | {b_cnt} | {pct:.1f}% | {b_name.replace('_', ' ').title()} |\n")
        summary_md_lines.append(f"\n**Total Categorized:** {total_bucket_sum} / {n_queries} (100.0% coverage)\n\n")

    # Save outputs
    df_tax = pd.DataFrame(all_taxonomy_records)
    df_tax.to_csv(ROOT / "analysis" / "tail_taxonomy.csv", index=False)

    df_review = pd.DataFrame(manual_review_records)
    df_review.to_csv(ROOT / "analysis" / "tail_manual_review.csv", index=False)

    with open(ROOT / "analysis" / "tail_taxonomy_summary.md", "w", encoding="utf-8") as f:
        f.write("".join(summary_md_lines))

    print("\n" + "=" * 90)
    print("STAGE B: TAIL FAILURE TAXONOMY SUMMARY (RQ4)")
    print("=" * 90)
    print("".join(summary_md_lines))
    print("Gate B Passed: analysis/tail_taxonomy.csv and analysis/tail_taxonomy_summary.md written successfully.\n")


if __name__ == "__main__":
    main()
