#!/usr/bin/env python3
"""
Official XOR-Retrieve Scorer for Table 6 (RQ3):
- Dense Single Channel (Top-100 & Top-200)
- 6-Channel RRF (Top-100 & Top-200)
- Reranked Baseline (R-Dense Top-100)
- Reranked Six-Channel (R-Union100 & R-Union200)
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
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

def score_token_xor(dids, doc_map, ans_list, token_cache):
    accum_tokens = []
    hit_2 = 0
    hit_5 = 0
    for did in dids:
        if did not in token_cache:
            token_cache[did] = word_tokenize(doc_map.get(did, ""))
        accum_tokens.extend(token_cache[did])
        if len(accum_tokens) >= 2000 and hit_2 == 0:
            if any(a in " ".join(accum_tokens[:2000]) for a in ans_list):
                hit_2 = 1
        if len(accum_tokens) >= 5000:
            if any(a in " ".join(accum_tokens[:5000]) for a in ans_list):
                hit_5 = 1
            break
    else:
        if hit_2 == 0 and any(a in " ".join(accum_tokens[:2000]) for a in ans_list):
            hit_2 = 1
        if hit_5 == 0 and any(a in " ".join(accum_tokens[:5000]) for a in ans_list):
            hit_5 = 1
    return hit_2, hit_5

def compute_rrf_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores = defaultdict(float)
    for r_list in ranked_lists:
        for rank, did in enumerate(r_list, start=1):
            scores[did] += 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

def main():
    print("="*80)
    print("RQ3: OFFICIAL TOKEN EVALUATION (TABLE 6)")
    print("="*80)

    token_cache = {}

    for track in ["bn", "te"]:
        queries = load_queries(track)
        doc_map = load_doc_text_map(track)
        runs_dir = ROOT / "runs" / track

        # Load channel runs
        c1 = runs_dir / f"S1__{track}.trec"
        c2 = runs_dir / f"S3__{track}.trec"
        c3 = runs_dir / f"S2__{track}.trec"
        c4 = runs_dir / f"S4__{track}.trec" # fallback sparse/structured
        c5 = runs_dir / f"S6__{track}.trec"
        c6 = runs_dir / f"S8__{track}.trec"

        def load_run(p):
            d = defaultdict(list)
            if not p.exists(): return d
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        d[parts[0]].append((int(parts[3]), parts[2]))
            return {k: [did for _, did in sorted(v)] for k, v in d.items()}

        r_s1 = load_run(c1)
        r_s3 = load_run(c2)
        r_s2 = load_run(c3)

        base_run = r_s3 if track == "bn" else r_s1
        base_name = "Dense Single Channel (S3 bn / S1 te)"

        # Let's inspect step9_six_channel_fixed rankings if saved or recompute RRF
        valid_queries = []
        for q in queries:
            ans = [a for a in q.get("answers", []) if a not in ["yes", "no"]]
            if ans:
                valid_queries.append((q, ans))

        # Evaluate Dense top-100 & top-200
        for top_n in [100, 200]:
            h2, h5, h_cov = [], [], []
            for q, ans in valid_queries:
                qid = str(q["id"])
                dids = base_run.get(qid, [])[:top_n]
                hit2, hit5 = score_token_xor(dids, doc_map, ans, token_cache)
                cov = 1 if any(any(a in doc_map.get(d, "") for a in ans) for d in dids) else 0
                h2.append(hit2)
                h5.append(hit5)
                h_cov.append(cov)
            print(f"[{track.upper()}] {base_name} (Top-{top_n}): R@2kt={np.mean(h2)*100:.2f}%, R@5kt={np.mean(h5)*100:.2f}%, Cov={np.mean(h_cov)*100:.2f}%")

if __name__ == "__main__":
    main()
