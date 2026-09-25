#!/usr/bin/env python3
"""
Official XOR-Retrieve Scorer across ALL paper tables:
- Table 3: RQ1 Granularity ladder
- Table 4: M1 views & LSE
- Table 5: RQ2 Multilingual encoders (BGE-M3 vs mE5-small)
- Table 6: RQ3 Matched candidate budgets & reranking
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
from sentence_transformers import SentenceTransformer
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

def get_local_model_path(model_id: str) -> str:
    m_name = model_id.replace("/", "--")
    cache_hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{m_name}" / "snapshots"
    if cache_hub.exists():
        for d in cache_hub.iterdir():
            if d.is_dir() and (d / "config.json").exists() and ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()):
                return str(d)
    return model_id

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

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    token_cache = {}
    
    # ---------------------------------------------------------
    # RQ2: mE5-small encoder ladder
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("RQ2: mE5-small OFFICIAL TOKEN EVALUATION")
    print("="*80)
    e5_path = get_local_model_path("intfloat/multilingual-e5-small")
    e5_model = SentenceTransformer(e5_path, device=device)
    
    me5_results = []
    for track in ["bn", "te"]:
        queries = load_queries(track)
        doc_map = load_doc_text_map(track)
        doc_ids = list(doc_map.keys())
        passages = [doc_map[did] for did in doc_ids]

        # Encode passages
        print(f"Encoding {len(passages)} passages with mE5-small ({track})...")
        p_texts = [f"passage: {p}" for p in passages]
        p_embs = e5_model.encode(p_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)

        # Evaluate S1 (Native) and S3 (MT)
        llm_data_path = ROOT / "data" / "processed" / f"{track}_llm.jsonl"
        q_map = {}
        with open(llm_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    q_map[str(o["id"])] = o

        for sys_id, q_key in [("S1 (Native)", "question"), ("S3 (MT)", "translation")]:
            q_texts = []
            valid_queries = []
            for q in queries:
                qid = str(q["id"])
                ans_list = [a for a in q.get("answers", []) if a not in ["yes", "no"]]
                if not ans_list:
                    continue
                valid_queries.append((q, ans_list))
                if q_key == "question":
                    q_texts.append(f"query: {q['question']}")
                else:
                    t_str = q_map.get(qid, {}).get("translation", q["question"])
                    q_texts.append(f"query: {t_str}")

            q_embs = e5_model.encode(q_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
            sims = np.dot(q_embs, p_embs.T)

            hits_2, hits_5 = [], []
            for i, (q, ans_list) in enumerate(valid_queries):
                s = sims[i]
                top_idx = np.argpartition(s, -100)[-100:]
                sorted_idx = top_idx[np.argsort(-s[top_idx])]
                top_dids = [doc_ids[idx] for idx in sorted_idx]
                h2, h5 = score_token_xor(top_dids, doc_map, ans_list, token_cache)
                hits_2.append(h2)
                hits_5.append(h5)

            me5_results.append({
                "Track": track,
                "Model": "mE5-small",
                "System": sys_id,
                "R@2kt": f"{np.mean(hits_2)*100:.2f}%",
                "R@5kt": f"{np.mean(hits_5)*100:.2f}%"
            })

    print(pd.DataFrame(me5_results).to_string(index=False))

if __name__ == "__main__":
    main()
