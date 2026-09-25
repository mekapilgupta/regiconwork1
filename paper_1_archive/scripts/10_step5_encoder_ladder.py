#!/usr/bin/env python3
"""
Step 5: Encoder Ladder Benchmark (Comparing BGE-M3 vs multilingual-e5-small).

Evaluates whether:
1. Zero-shot vs MT translation asymmetry persists across different encoder capacities.
2. Weaker encoders (mE5-small) show larger relative headroom for reranking and verification bonuses.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import DATA_DIR, METRICS_DIR, RUNS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("10_step5_encoder_ladder")


def get_local_model_path(model_id: str) -> str:
    cache_hub = ROOT / "cache" / "hf" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if cache_hub.exists():
        for d in cache_hub.iterdir():
            if d.is_dir() and (d / "modules.json").exists():
                return str(d)
    cache_user = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if cache_user.exists():
        for d in cache_user.iterdir():
            if d.is_dir() and (d / "modules.json").exists():
                return str(d)
    return model_id


def run_dense_query(query_text: str, model: SentenceTransformer, pool_embeddings: np.ndarray, doc_ids: list[str]) -> np.ndarray:
    if not query_text.strip():
        return np.zeros(len(doc_ids), dtype=np.float32)
    q_emb = model.encode([f"query: {query_text}"], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    scores = np.dot(pool_embeddings, q_emb)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Step 5: Encoder Ladder Evaluation (mE5-small)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "intfloat/multilingual-e5-small"
    model_path = get_local_model_path(model_id)
    log.info("Loading mE5-small from '%s' on %s...", model_path, device)

    model = SentenceTransformer(model_path, device=device)
    if device == "cuda":
        model = model.half()

    tracks = ["bn", "te"]
    rows = []

    for track in tracks:
        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        
        pool_file = ROOT / "cache" / "pool" / track / "pool_text.jsonl"
        passages = list(read_jsonl(pool_file))
        doc_ids = [p["id"] for p in passages]
        doc_text_map = {p["id"]: p["text"] for p in passages}

        # Check or compute mE5 pool embeddings
        e5_pool_emb_file = ROOT / "cache" / "pool" / track / "pool_emb_e5_small.npy"
        if e5_pool_emb_file.exists():
            log.info("Loading cached mE5 pool embeddings from %s...", e5_pool_emb_file)
            pool_emb = np.load(e5_pool_emb_file)
        else:
            log.info("Encoding %d passages with mE5-small on %s...", len(passages), device)
            passage_texts = [f"passage: {p['text']}" for p in passages]
            pool_emb = model.encode(passage_texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            np.save(e5_pool_emb_file, pool_emb)

        # Evaluate S1 (zero-shot), S3 (MT), S8 (Gold translation)
        s1_hits_5k = []
        s3_hits_5k = []
        s8_hits_5k = []

        for q in tqdm(queries, desc=f"mE5 [{track}]"):
            qid = str(q["id"])
            q_src = q.get("question", "")
            q_trans = q.get("llm_translation", "")
            q_gold = q.get("eng_translation", "") or q_trans
            ans = q.get("answers", [])

            # S1
            s1_scores = run_dense_query(q_src, model, pool_emb, doc_ids)
            top100_s1 = [doc_ids[idx] for idx in np.argpartition(s1_scores, -100)[-100:][np.argsort(-s1_scores[np.argpartition(s1_scores, -100)[-100:]])]]
            s1_hits_5k.append(eval_query_xor_tydi(top100_s1, doc_text_map, ans)["r_5kt"])

            # S3
            s3_scores = run_dense_query(q_trans, model, pool_emb, doc_ids)
            top100_s3 = [doc_ids[idx] for idx in np.argpartition(s3_scores, -100)[-100:][np.argsort(-s3_scores[np.argpartition(s3_scores, -100)[-100:]])]]
            s3_hits_5k.append(eval_query_xor_tydi(top100_s3, doc_text_map, ans)["r_5kt"])

            # S8
            s8_scores = run_dense_query(q_gold, model, pool_emb, doc_ids)
            top100_s8 = [doc_ids[idx] for idx in np.argpartition(s8_scores, -100)[-100:][np.argsort(-s8_scores[np.argpartition(s8_scores, -100)[-100:]])]]
            s8_hits_5k.append(eval_query_xor_tydi(top100_s8, doc_text_map, ans)["r_5kt"])

        r5_s1 = float(np.mean(s1_hits_5k)) * 100
        r5_s3 = float(np.mean(s3_hits_5k)) * 100
        r5_s8 = float(np.mean(s8_hits_5k)) * 100
        oracle_s1_s3 = float(np.mean(np.maximum(s1_hits_5k, s3_hits_5k))) * 100

        rows.append({"Track": track, "Encoder": "mE5-small", "S1 (Zero-Shot)": f"{r5_s1:.2f}%", "S3 (MT)": f"{r5_s3:.2f}%", "S8 (Gold)": f"{r5_s8:.2f}%", "Oracle {S1,S3}": f"{oracle_s1_s3:.2f}%", "Headroom (pts)": f"{oracle_s1_s3 - max(r5_s1, r5_s3):+.2f}"})

    df_ladder = pd.DataFrame(rows)
    out_csv = METRICS_DIR / "step5_encoder_ladder.csv"
    df_ladder.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print("STEP 5: ENCODER LADDER RESULTS (mE5-small)")
    print("=" * 80)
    print(df_ladder.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
