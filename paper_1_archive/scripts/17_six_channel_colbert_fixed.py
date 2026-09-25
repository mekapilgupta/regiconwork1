#!/usr/bin/env python3
"""
Step 9: Fixed True Six-Channel Native Hybrid (Dense + Sparse + True ColBERT MaxSim) and Union Reranking.

Implements:
1. True ColBERT MaxSim Late-Interaction:
   MaxSim(q, d) = sum_{i in q} max_{j in d} (v_{q,i} . v_{d,j})
   Computed via GPU FP16 matrix operations over candidate token tensors.
2. Inverted-Index Sparse Channel using BGE-M3 lexical head weights.
3. Dense Channels C1 (q_src) and C2 (q_mt).
4. Full 6-Channel RRF Fusion (k=60) over:
   - Top-100 Pool
   - Top-200 Union Pool
5. Deep Cross-Encoder Reranker (bge-reranker-v2-m3) on:
   - Baseline R-S3 (bn) / R-S1 (te)
   - Fused 6-Channel Top-100
   - Fused 6-Channel Top-200
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Pre-set Java, dummy OpenAI, offline flags
os.environ["OPENAI_API_KEY"] = "dummy"
os.environ.setdefault("JAVA_HOME", "/home/kshitij22256/miniconda3/envs/clir_cuda124")
os.environ.setdefault("JVM_PATH", "/home/kshitij22256/miniconda3/envs/clir_cuda124/lib/server/libjvm.so")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.config import DATA_DIR, METRICS_DIR, RUNS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("17_six_channel_fixed")


def get_local_model_path(model_id: str) -> str:
    m_name = model_id.replace("/", "--")
    cache_hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{m_name}" / "snapshots"
    if cache_hub.exists():
        for d in cache_hub.iterdir():
            if d.is_dir() and (d / "config.json").exists() and ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()):
                return str(d)
    repo_hub = ROOT / "cache" / "hf" / f"models--{m_name}" / "snapshots"
    if repo_hub.exists():
        for d in repo_hub.iterdir():
            if d.is_dir() and (d / "config.json").exists() and ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()):
                return str(d)
    return model_id


def load_doc_text_map(track: str) -> dict[str, str]:
    pool_file = ROOT / "cache" / "pool" / track / "pool_text.jsonl"
    passages = list(read_jsonl(pool_file))
    return {p["id"]: p["text"] for p in passages}


def compute_rrf_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores = defaultdict(float)
    for r_list in ranked_lists:
        for rank, did in enumerate(r_list, start=1):
            scores[did] += 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Fixed Six-Channel Native Hybrid & Union Reranking")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info("Running on device: %s (FP16 Accelerated)", device)

    # 1. Load BGE-M3 base transformer + colbert + sparse heads
    m3_snapshot = get_local_model_path("BAAI/bge-m3")
    log.info("Loading BGE-M3 from '%s'...", m3_snapshot)
    tokenizer = AutoTokenizer.from_pretrained(m3_snapshot)
    base_model = AutoModel.from_pretrained(m3_snapshot).to(device)
    if device.startswith("cuda"):
        base_model = base_model.half()
    base_model.eval()

    # Find linear heads
    repo_snapshot = ROOT / "cache" / "hf" / "models--BAAI--bge-m3" / "snapshots"
    head_dir = Path(m3_snapshot)
    if repo_snapshot.exists():
        for d in repo_snapshot.iterdir():
            if (d / "colbert_linear.pt").exists():
                head_dir = d
                break

    log.info("Loading ColBERT & Sparse heads from '%s'...", head_dir)
    colbert_linear = torch.nn.Linear(1024, 1024).to(device)
    sparse_linear = torch.nn.Linear(1024, 1).to(device)
    colbert_linear.load_state_dict(torch.load(head_dir / "colbert_linear.pt", map_location=device, weights_only=True))
    sparse_linear.load_state_dict(torch.load(head_dir / "sparse_linear.pt", map_location=device, weights_only=True))
    if device.startswith("cuda"):
        colbert_linear = colbert_linear.half()
        sparse_linear = sparse_linear.half()
    colbert_linear.eval()
    sparse_linear.eval()

    # 2. Load Reranker
    reranker_snapshot = get_local_model_path("BAAI/bge-reranker-v2-m3")
    log.info("Loading bge-reranker-v2-m3 from '%s'...", reranker_snapshot)
    rerank_tok = AutoTokenizer.from_pretrained(reranker_snapshot)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_snapshot).to(device)
    if device.startswith("cuda"):
        rerank_model = rerank_model.half()
    rerank_model.eval()

    tracks = ["bn", "te"]
    master_results = []

    for track in tracks:
        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        n_queries = len(queries)
        doc_text_map = load_doc_text_map(track)

        pool_passages = list(read_jsonl(ROOT / "cache" / "pool" / track / "pool_text.jsonl"))
        doc_ids = [p["id"] for p in pool_passages]
        n_docs = len(doc_ids)

        log.info("=" * 80)
        log.info("TRACK %s: EVALUATING TRUE 6-CHANNEL HYBRID (N=%d queries, %d pool docs)", track.upper(), n_queries, n_docs)
        log.info("=" * 80)

        # Dense pool embeddings
        bge_pool_emb = np.load(ROOT / "cache" / "pool" / track / "pool_emb.npy")

        # Inverted index for fast sparse matching
        log.info("Building in-memory inverted index for %d passages...", n_docs)
        inverted_index = defaultdict(list)
        for d_i, p in enumerate(pool_passages):
            words = set(re.findall(r"\w+", p["text"].lower()))
            for w in words:
                inverted_index[w].append(d_i)

        # Encode queries for all 3 modes
        def encode_query_batch(texts: list[str]):
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = base_model(**inputs).last_hidden_state
                d_vecs = F.normalize(out[:, 0], p=2, dim=-1).float().cpu().numpy()
                sp_weights = F.relu(sparse_linear(out)).squeeze(-1).float().cpu().numpy()
                col_vecs = F.normalize(colbert_linear(out), p=2, dim=-1)  # keep on device for MaxSim
                input_ids = inputs["input_ids"].cpu().numpy()
            return d_vecs, sp_weights, col_vecs, input_ids

        q_src_texts = [q.get("question", "").strip() for q in queries]
        q_mt_texts = [(q.get("llm_translation") or "").strip() for q in queries]

        d_src, sp_src, col_src, ids_src = encode_query_batch(q_src_texts)
        d_mt, sp_mt, col_mt, ids_mt = encode_query_batch(q_mt_texts)

        # Precompute Dense Scores
        dense_src_scores = np.dot(bge_pool_emb, d_src.T).T  # (n_q, n_d)
        dense_mt_scores = np.dot(bge_pool_emb, d_mt.T).T

        # Sparse scoring function
        def get_sparse_ranked_list(q_tokens, q_weights, top_k=200):
            token_dict = {tid: w for tid, w in zip(q_tokens, q_weights) if w > 0.05 and tid not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]}
            if not token_dict:
                return doc_ids[:top_k]
            scores = np.zeros(n_docs, dtype=np.float32)
            for tid, w in token_dict.items():
                tok_str = tokenizer.decode([tid]).strip().lower()
                if len(tok_str) > 1 and tok_str in inverted_index:
                    for d_i in inverted_index[tok_str]:
                        scores[d_i] += w
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            sorted_top = top_idx[np.argsort(-scores[top_idx])]
            return [doc_ids[i] for i in sorted_top]

        log.info("Computing Channel Rankings for track '%s'...", track)
        c1_dense_src = {}
        c2_dense_mt = {}
        c3_sparse_src = {}
        c4_sparse_mt = {}
        c5_colbert_src = {}
        c6_colbert_mt = {}

        for q_i, q in enumerate(tqdm(queries, desc=f"Channels [{track}]")):
            qid = str(q["id"])

            # C1: Dense native
            s_c1 = dense_src_scores[q_i]
            top_c1 = np.argpartition(s_c1, -200)[-200:][np.argsort(-s_c1[np.argpartition(s_c1, -200)[-200:]])]
            c1_dense_src[qid] = [doc_ids[i] for i in top_c1]

            # C2: Dense MT
            s_c2 = dense_mt_scores[q_i]
            top_c2 = np.argpartition(s_c2, -200)[-200:][np.argsort(-s_c2[np.argpartition(s_c2, -200)[-200:]])]
            c2_dense_mt[qid] = [doc_ids[i] for i in top_c2]

            # C3: Sparse native & C4: Sparse MT
            c3_sparse_src[qid] = get_sparse_ranked_list(ids_src[q_i], sp_src[q_i], top_k=200)
            c4_sparse_mt[qid] = get_sparse_ranked_list(ids_mt[q_i], sp_mt[q_i], top_k=200)

            # C5: ColBERT Native & C6: ColBERT MT
            # Compute real token MaxSim over top dense candidate pool
            cand_pool_dids = list(dict.fromkeys(c1_dense_src[qid][:100] + c2_dense_mt[qid][:100]))
            cand_texts = [doc_text_map.get(did, "") for did in cand_pool_dids]
            
            with torch.no_grad():
                inp_cands = tokenizer(cand_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                cand_out = base_model(**inp_cands).last_hidden_state
                cand_col_vecs = F.normalize(colbert_linear(cand_out), p=2, dim=-1)  # (B, L_d, 1024)

                # MaxSim for q_src: (B, L_d, 1024) @ (1024, L_q) -> (B, L_d, L_q)
                q_mat_src = col_src[q_i].transpose(0, 1)  # (1024, L_q)
                sim_matrix_src = torch.matmul(cand_col_vecs, q_mat_src)  # (B, L_d, L_q)
                maxsim_src = sim_matrix_src.max(dim=1).values.sum(dim=-1).float().cpu().numpy()  # (B,)

                # MaxSim for q_mt
                q_mat_mt = col_mt[q_i].transpose(0, 1)  # (1024, L_q)
                sim_matrix_mt = torch.matmul(cand_col_vecs, q_mat_mt)  # (B, L_d, L_q)
                maxsim_mt = sim_matrix_mt.max(dim=1).values.sum(dim=-1).float().cpu().numpy()  # (B,)

            c5_colbert_src[qid] = [cand_pool_dids[idx] for idx in np.argsort(-maxsim_src)]
            c6_colbert_mt[qid] = [cand_pool_dids[idx] for idx in np.argsort(-maxsim_mt)]

        # -------------------------------------------------------------
        # 6-Channel Fusion Lists
        # -------------------------------------------------------------
        fused_6ch_100 = {}
        fused_6ch_200 = {}

        for q in queries:
            qid = str(q["id"])
            fused = compute_rrf_fusion([
                c1_dense_src[qid], c2_dense_mt[qid],
                c3_sparse_src[qid], c4_sparse_mt[qid],
                c5_colbert_src[qid], c6_colbert_mt[qid]
            ], k=60)
            fused_6ch_100[qid] = fused[:100]
            fused_6ch_200[qid] = fused[:200]

        def evaluate_list_dict(name: str, rank_dict: dict[str, list[str]]):
            met_list = []
            for q in queries:
                qid = str(q["id"])
                ans = q.get("answers", [])
                dids = rank_dict[qid]
                met_list.append(eval_query_xor_tydi(dids, doc_text_map, ans))
            r2 = float(np.mean([m["r_2kt"] for m in met_list])) * 100
            r5 = float(np.mean([m["r_5kt"] for m in met_list])) * 100
            cfr = float(np.mean([m["catastrophic_failure"] for m in met_list])) * 100
            r100 = 100.0 - cfr
            return {"Track": track, "System": name, "R@2kt": f"{r2:.2f}%", "R@5kt": f"{r5:.2f}%", "R@100": f"{r100:.2f}%", "CFR": f"{cfr:.2f}%"}

        base_channel = c2_dense_mt if track == "bn" else c1_dense_src
        base_name = "S3 (MT Dense)" if track == "bn" else "S1 (Zero-Shot Dense)"
        
        master_results.append(evaluate_list_dict(f"Baseline: {base_name}", base_channel))
        master_results.append(evaluate_list_dict("Six-Channel Hybrid (Top-100 Pool)", fused_6ch_100))
        master_results.append(evaluate_list_dict("Six-Channel Hybrid (Top-200 Pool)", fused_6ch_200))

        # -------------------------------------------------------------
        # Cross-Encoder Reranking
        # -------------------------------------------------------------
        log.info("Reranking 6-Channel Candidate Pools with bge-reranker-v2-m3...")

        # Collect unique pairs across baseline top-100, 6ch top-100, 6ch top-200
        unique_pairs = []
        for q in queries:
            qid = str(q["id"])
            q_txt = (q.get("llm_translation") or q.get("question") or "").strip()
            all_dids = set(base_channel[qid][:100] + fused_6ch_200[qid][:200])
            for did in all_dids:
                doc_txt = doc_text_map.get(did, "")
                unique_pairs.append((qid, did, q_txt, doc_txt))

        log.info("Scoring %d unique pairs with FP16 reranker...", len(unique_pairs))
        rerank_score_map = {}
        with torch.no_grad():
            for b_i in range(0, len(unique_pairs), args.batch_size):
                b_chunk = unique_pairs[b_i : b_i + args.batch_size]
                inp = rerank_tok([p[2] for p in b_chunk], [p[3] for p in b_chunk], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                if device.startswith("cuda"):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = rerank_model(**inp).logits.view(-1).float().cpu().numpy()
                else:
                    logits = rerank_model(**inp).logits.view(-1).float().cpu().numpy()
                for p, sc in zip(b_chunk, logits):
                    rerank_score_map[(p[0], p[1])] = float(sc)

        def rerank_ranking_dict(name: str, rank_dict: dict[str, list[str]], top_n: int = 100):
            reranked_dict = {}
            for q in queries:
                qid = str(q["id"])
                dids = rank_dict[qid][:top_n]
                sorted_dids = sorted(dids, key=lambda d: rerank_score_map.get((qid, d), -999.0), reverse=True)
                reranked_dict[qid] = sorted_dids
            return evaluate_list_dict(name, reranked_dict)

        master_results.append(rerank_ranking_dict(f"Reranked Baseline: R-{base_name[:2]} (Top-100)", base_channel, top_n=100))
        master_results.append(rerank_ranking_dict("Reranked Six-Channel (Top-100)", fused_6ch_100, top_n=100))
        master_results.append(rerank_ranking_dict("Reranked Six-Channel (Top-200)", fused_6ch_200, top_n=200))

    df_out = pd.DataFrame(master_results)
    out_csv = METRICS_DIR / "step9_six_channel_fixed.csv"
    df_out.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("STEP 9: TRUE SIX-CHANNEL HYBRID (WITH REAL COLBERT MAXSIM) & RERANKING")
    print("=" * 90)
    print(df_out.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
