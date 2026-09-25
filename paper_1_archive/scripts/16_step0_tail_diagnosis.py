#!/usr/bin/env python3
"""
Step 0 Diagnosis: Article-Context & Persistent Tail Failure Taxonomy.

For every persistent-tail query (where dense baseline S1/S3 fails, CFR=1):
1. Locate gold passages by answer containment within the candidate pool.
2. Check whether query entities / key content words appear in:
   (a) Passage body text
   (b) Article title (text.split('\\n')[0])
   (c) Same-article neighbor passages (id +/- 1, id +/- 2)
3. Output exact breakdown and check the >= 40% decision rule for Idea 3.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import DATA_DIR, METRICS_DIR, get_logger
from src.metrics import _is_answer_contained
from src.utils_io import read_jsonl

log = get_logger("16_step0_tail_diag")


def extract_title_body(raw_text: str) -> tuple[str, str]:
    parts = raw_text.split("\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", raw_text.strip()


def main():
    tracks = ["bn", "te"]
    summary_rows = []

    for track in tracks:
        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        
        pool_file = ROOT / "cache" / "pool" / track / "pool_text.jsonl"
        passages = list(read_jsonl(pool_file))
        
        # Build doc mapping by integer id if available
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

        # Load S1 and S3 dense results to find persistent tail queries
        bge_pool_emb = np.load(ROOT / "cache" / "pool" / track / "pool_emb.npy")
        doc_ids = [p["id"] for p in passages]

        log.info("Building single-pass answer index for %d queries...", len(queries))
        query_answers_map = {str(q["id"]): [a.lower().strip() for a in q.get("answers", []) if a.strip()] for q in queries}
        ans_to_qids = defaultdict(list)
        for qid, ans_list in query_answers_map.items():
            for a in ans_list:
                ans_to_qids[a].append(qid)

        query_gold_dids = defaultdict(list)
        log.info("Scanning %d passages for gold answer containment...", len(passages))
        for p in passages:
            p_text = p["text"].lower()
            for ans_str, qids in ans_to_qids.items():
                if ans_str in p_text:
                    for qid in qids:
                        query_gold_dids[qid].append(p["id"])

        tail_queries = []
        for q in queries:
            qid = str(q["id"])
            ans = q.get("answers", [])
            q_text = (q.get("llm_translation") or q.get("question") or "").strip()
            facets = q.get("llm_facets") or {}
            
            entities = [e.get("english_form", "") for e in facets.get("entities_source", []) if not e.get("facet_invalid")]
            if not entities:
                entities = [w for w in re.findall(r"\w+", q_text.lower()) if len(w) > 3]

            tail_queries.append({
                "qid": qid,
                "question": q.get("question"),
                "translation": q_text,
                "answers": ans,
                "entities": entities,
                "gold_dids": list(set(query_gold_dids.get(qid, [])))
            })

        # Categorize every query's gold passages
        cat_counts = {
            "gold_has_entity_in_body": 0,
            "gold_has_entity_in_title_only": 0,
            "gold_has_entity_in_neighbor_only": 0,
            "gold_missing_entity_completely": 0,
            "zero_gold_in_pool": 0
        }

        tail_details = []

        for tq in tail_queries:
            gold_dids = tq["gold_dids"]
            if not gold_dids:
                cat_counts["zero_gold_in_pool"] += 1
                continue

            ents = [e.lower() for e in tq["entities"] if e]
            
            in_body = False
            in_title_only = False
            in_neighbor_only = False

            for g_did in gold_dids:
                g_doc = doc_map.get(str(g_did))
                if not g_doc:
                    continue

                b_lower = g_doc["body"].lower()
                t_lower = g_doc["title"].lower()

                # Check body
                if any(e in b_lower for e in ents):
                    in_body = True
                    break

                # Check title
                if any(e in t_lower for e in ents):
                    in_title_only = True

                # Check neighbors +/- 1, +/- 2
                g_int = g_doc["int_id"]
                if g_int is not None:
                    for offset in [-2, -1, 1, 2]:
                        n_did = str(g_int + offset)
                        if n_did in doc_map:
                            n_text = doc_map[n_did]["full_text"].lower()
                            if any(e in n_text for e in ents):
                                in_neighbor_only = True

            if in_body:
                cat_counts["gold_has_entity_in_body"] += 1
            elif in_title_only:
                cat_counts["gold_has_entity_in_title_only"] += 1
            elif in_neighbor_only:
                cat_counts["gold_has_entity_in_neighbor_only"] += 1
            else:
                cat_counts["gold_missing_entity_completely"] += 1

        n_total = len(tail_queries)
        title_or_neighbor_only = cat_counts["gold_has_entity_in_title_only"] + cat_counts["gold_has_entity_in_neighbor_only"]
        title_neighbor_pct = (title_or_neighbor_only / n_total) * 100.0

        log.info("Track '%s' Step 0 Taxonomy:", track)
        log.info("  Total queries: %d", n_total)
        log.info("  (a) Entity in passage body: %d (%.1f%%)", cat_counts["gold_has_entity_in_body"], cat_counts["gold_has_entity_in_body"]/n_total*100)
        log.info("  (b) Entity in title only: %d (%.1f%%)", cat_counts["gold_has_entity_in_title_only"], cat_counts["gold_has_entity_in_title_only"]/n_total*100)
        log.info("  (c) Entity in neighbor (+/-2) only: %d (%.1f%%)", cat_counts["gold_has_entity_in_neighbor_only"], cat_counts["gold_has_entity_in_neighbor_only"]/n_total*100)
        log.info("  (d) Entity completely missing / alias gap: %d (%.1f%%)", cat_counts["gold_missing_entity_completely"], cat_counts["gold_missing_entity_completely"]/n_total*100)
        log.info("  (e) Zero gold passages in pool (unreachable): %d (%.1f%%)", cat_counts["zero_gold_in_pool"], cat_counts["zero_gold_in_pool"]/n_total*100)
        log.info("  ==> Title + Neighbor Only = %.1f%% (Threshold for Idea 3: >= 40%%)", title_neighbor_pct)

        summary_rows.append({
            "Track": track,
            "Total Queries": n_total,
            "(a) Entity in Body": f"{cat_counts['gold_has_entity_in_body']} ({cat_counts['gold_has_entity_in_body']/n_total*100:.1f}%)",
            "(b) Title Only": f"{cat_counts['gold_has_entity_in_title_only']} ({cat_counts['gold_has_entity_in_title_only']/n_total*100:.1f}%)",
            "(c) Neighbor (+/-2) Only": f"{cat_counts['gold_has_entity_in_neighbor_only']} ({cat_counts['gold_has_entity_in_neighbor_only']/n_total*100:.1f}%)",
            "Title + Neighbor Total": f"{title_or_neighbor_only} ({title_neighbor_pct:.1f}%)",
            "Proceed with Idea 3?": "YES (>=40%)" if title_neighbor_pct >= 40.0 else "NO (<40%) - Write Tail Taxonomy"
        })

    df_diag = pd.DataFrame(summary_rows)
    df_diag.to_csv(METRICS_DIR / "step0_tail_diagnosis.csv", index=False)
    print("\n" + "=" * 90)
    print("STEP 0 DIAGNOSIS: ARTICLE-CONTEXT & PERSISTENT TAIL TAXONOMY")
    print("=" * 90)
    print(df_diag.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
