#!/usr/bin/env python3
"""
Official XOR-Retrieve Scorer: Token-budget evaluation (R@2kt and R@5kt).
Follows exact official implementation from AkariAsai/XORQA evals/eval_xor_retrieve.py.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

ROOT = Path(__file__).resolve().parent.parent

def load_queries(track: str):
    q_file = ROOT / "data" / "filtered" / f"xor_{track}.jsonl"
    queries = []
    with open(q_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries

def load_doc_text_map(track: str):
    p_file = ROOT / "cache" / "pool" / track / "pool_text.jsonl"
    doc_map = {}
    with open(p_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                doc_map[str(obj["id"])] = obj.get("text", "")
    return doc_map

def load_trec_run(run_file: Path):
    run_dict = defaultdict(list)
    with open(run_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, did, rank = parts[0], parts[1], parts[2], int(parts[3])
                run_dict[qid].append((rank, did))
    # sort by rank
    final_dict = {}
    for qid, items in run_dict.items():
        items.sort(key=lambda x: x[0])
        final_dict[qid] = [did for _, did in items]
    return final_dict

def main():
    print("=" * 80)
    print("RUNNING OFFICIAL XOR-RETRIEVE TOKEN-BUDGET SCORER")
    print("=" * 80)

    for track in ["bn", "te"]:
        queries = load_queries(track)
        doc_map = load_doc_text_map(track)
        runs_dir = ROOT / "runs" / track

        print(f"\nEvaluating Track: {track.upper()} ({len(queries)} queries)...")

        token_cache = {}
        def get_tokens(did):
            if did not in token_cache:
                text = doc_map.get(did, "")
                token_cache[did] = word_tokenize(text)
            return token_cache[did]

        systems = [
            ("S1", "Native Whole Question", runs_dir / f"S1__{track}.trec"),
            ("S3", "Machine Translated Question", runs_dir / f"S3__{track}.trec"),
            ("S8", "Human Reference Translation", runs_dir / f"S8__{track}.trec"),
            ("S4", "Concatenated Facets", runs_dir / f"S4__{track}.trec"),
            ("S6", "Multi-Vector Facets", runs_dir / f"S6__{track}.trec"),
            ("S9", "Arbitrary 3-Spans", runs_dir / f"S9__{track}.trec"),
            ("S5", "Whole Question + Semantic Tags", runs_dir / f"S5__{track}.trec"),
            ("S5r", "Whole Question + Random Words", runs_dir / f"S5r__{track}.trec"),
        ]

        results = []
        for sys_id, sys_name, r_path in systems:
            if not r_path.exists():
                print(f"Skipping {sys_id}: {r_path} not found")
                continue

            run_dict = load_trec_run(r_path)

            hits_2k = []
            hits_5k = []
            r100_hits = []

            for q in queries:
                qid = str(q["id"])
                ans_list = [a for a in q.get("answers", []) if a not in ["yes", "no"]]
                if not ans_list:
                    continue

                dids = run_dict.get(qid, [])

                # R@100
                hit_100 = 0
                for did in dids[:100]:
                    txt = doc_map.get(did, "")
                    if any(a in txt for a in ans_list):
                        hit_100 = 1
                        break
                r100_hits.append(hit_100)

                # Token budget 2kt and 5kt
                accum_tokens = []
                hit_2 = 0
                hit_5 = 0

                for did in dids:
                    toks = get_tokens(did)
                    accum_tokens.extend(toks)
                    if len(accum_tokens) >= 2000 and hit_2 == 0:
                        concat_str = " ".join(accum_tokens[:2000])
                        if any(a in concat_str for a in ans_list):
                            hit_2 = 1
                    if len(accum_tokens) >= 5000:
                        concat_str = " ".join(accum_tokens[:5000])
                        if any(a in concat_str for a in ans_list):
                            hit_5 = 1
                        break
                else:
                    # In case total tokens < 5000
                    if hit_2 == 0:
                        concat_str = " ".join(accum_tokens[:2000])
                        if any(a in concat_str for a in ans_list):
                            hit_2 = 1
                    if hit_5 == 0:
                        concat_str = " ".join(accum_tokens[:5000])
                        if any(a in concat_str for a in ans_list):
                            hit_5 = 1

                hits_2k.append(hit_2)
                hits_5k.append(hit_5)

            results.append({
                "System": sys_id,
                "Name": sys_name,
                "R@2kt": f"{np.mean(hits_2k)*100:.2f}%",
                "R@5kt": f"{np.mean(hits_5k)*100:.2f}%",
                "R@100": f"{np.mean(r100_hits)*100:.2f}%"
            })

        df = pd.DataFrame(results)
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
