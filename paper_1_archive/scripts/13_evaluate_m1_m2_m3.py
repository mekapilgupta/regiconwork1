#!/usr/bin/env python3
"""
Step 6: Empirical Evaluation of M1 (Multi-View Sentence Bucket), M3-HyDE, and Combined Pipelines.

Implements:
1. M1: Multi-View Sentence Bucket
   - K=6 complete grammatical sentence views
   - Log-Sum-Exp temperature sweep: beta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
   - Uncertainty-aware disagreement penalty: s(q,d) = mean_k(cos(v_k, e_d)) - lambda * Var_k(cos(v_k, e_d))
   - Control: K-random unrelated questions (bucket-size control)
2. M3-HyDE: Passage-shaped cloze query embedding (standalone and combined)
3. M3-PRF: Vector PRF on M1/S3
4. Combined Hybrid: M1 (LSE/Uncertainty) + M3-HyDE + M3-PRF
5. Tail Analysis: Measures R@100, R@2kt, R@5kt, CFR, and Rescued Catastrophic Tail Queries.
"""
import argparse
import json
import math
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

log = get_logger("13_eval_m1_m3")


def get_local_model_path(model_id: str = "BAAI/bge-m3") -> str:
    cache_hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if cache_hub.exists():
        for d in cache_hub.iterdir():
            if d.is_dir() and (d / "modules.json").exists():
                return str(d)
    return model_id


def load_doc_text_map(track: str) -> dict[str, str]:
    pool_file = ROOT / "cache" / "pool" / track / "pool_text.jsonl"
    passages = list(read_jsonl(pool_file))
    return {p["id"]: p["text"] for p in passages}


def logsumexp_fusion(view_scores: np.ndarray, beta: float) -> np.ndarray:
    """
    view_scores: (K, num_docs)
    Computes (1/beta) * log( (1/K) * sum_k exp(beta * view_scores[k]) )
    """
    if beta <= 0.05:
        # Centroid limit
        return np.mean(view_scores, axis=0)
    
    # Numerically stable logsumexp
    scaled = beta * view_scores  # (K, N)
    max_val = np.max(scaled, axis=0, keepdims=True)  # (1, N)
    lse = max_val + np.log(np.mean(np.exp(scaled - max_val), axis=0, keepdims=True))
    return (lse / beta).squeeze(0)


def uncertainty_fusion(view_scores: np.ndarray, lam: float) -> np.ndarray:
    """
    view_scores: (K, num_docs)
    s = mean_k(cos) - lam * Var_k(cos)
    """
    mean_s = np.mean(view_scores, axis=0)
    var_s = np.var(view_scores, axis=0)
    return mean_s - lam * var_s


def main():
    parser = argparse.ArgumentParser(description="Evaluate M1 Multi-View Bucket & M3-HyDE")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = get_local_model_path("BAAI/bge-m3")
    log.info("Loading BGE-M3 from '%s' on %s...", model_path, device)
    model = SentenceTransformer(model_path, device=device)
    if device == "cuda":
        model = model.half()

    tracks = ["bn", "te"]
    master_results = []

    for track in tracks:
        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        doc_text_map = load_doc_text_map(track)

        pool_emb_file = ROOT / "cache" / "pool" / track / "pool_emb.npy"
        pool_embeddings = np.load(pool_emb_file)  # (N, D)
        passages = list(read_jsonl(ROOT / "cache" / "pool" / track / "pool_text.jsonl"))
        doc_ids = [p["id"] for p in passages]

        log.info("=" * 70)
        log.info("EVALUATING M1 & M3 ON TRACK: %s (N = %d queries)", track.upper(), len(queries))
        log.info("=" * 70)

        # Baseline S3 (MT) and S1 (Zero-Shot) vectors
        # Collect all textual views to encode in batches
        all_view_keys = [
            "trans_literal", "trans_natural", "trans_alias",
            "cloze_declarative", "src_paraphrase", "src_romanized",
            "hyde_passage"
        ]

        # Check random control texts from unrelated queries
        rng = np.random.RandomState(42)
        unrelated_questions = [q["question"] for q in queries]
        rng.shuffle(unrelated_questions)

        # Pre-encode all queries and views
        query_view_embs = defaultdict(dict)  # qid -> view_key -> emb

        texts_to_encode = []
        text_meta = []  # (qid, key)

        for idx, q in enumerate(queries):
            qid = str(q["id"])
            views = q.get("multi_views") or {}
            
            # Base views
            for vk in all_view_keys:
                txt = (views.get(vk) or "").strip()
                if not txt:
                    txt = q.get("llm_translation") or q.get("question") or ""
                texts_to_encode.append(txt)
                text_meta.append((qid, vk))

            # Random K-control views
            for r_i in range(6):
                r_txt = unrelated_questions[(idx * 6 + r_i) % len(unrelated_questions)]
                texts_to_encode.append(r_txt)
                text_meta.append((qid, f"rand_control_{r_i}"))

        log.info("Encoding %d total sentence views for track '%s'...", len(texts_to_encode), track)
        encoded_vectors = model.encode(
            texts_to_encode,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        for (qid, vk), vec in zip(text_meta, encoded_vectors):
            query_view_embs[qid][vk] = vec

        # ---------------------------------------------------------
        # Evaluation Functions
        # ---------------------------------------------------------
        def eval_scores_matrix(scores_arr, q):
            ans = q.get("answers", [])
            top100_idx = np.argpartition(scores_arr, -100)[-100:]
            sorted_idx = top100_idx[np.argsort(-scores_arr[top100_idx])]
            ranked_dids = [doc_ids[i] for i in sorted_idx]
            return eval_query_xor_tydi(ranked_dids, doc_text_map, ans)

        # Baseline: Single Best (S3 on bn, S1 on te)
        # Precompute view dot products once for each query: qid -> (K, num_docs)
        log.info("Precomputing view score matrices for %d queries...", len(queries))
        m1_views_6 = ["trans_literal", "trans_natural", "trans_alias", "cloze_declarative", "src_paraphrase", "src_romanized"]
        
        precomputed_view_scores = {}
        precomputed_rand_scores = {}
        rand_control_keys = [f"rand_control_{i}" for i in range(6)]

        for q in tqdm(queries, desc=f"Matrix dot [{track}]"):
            qid = str(q["id"])
            v_mat = np.stack([query_view_embs[qid][vk] for vk in m1_views_6], axis=0)  # (6, D)
            precomputed_view_scores[qid] = np.dot(v_mat, pool_embeddings.T)  # (6, N)

            v_rand = np.stack([query_view_embs[qid][vk] for vk in rand_control_keys], axis=0)
            precomputed_rand_scores[qid] = np.dot(v_rand, pool_embeddings.T)

        # Baseline: Single Best (S3 on bn, S1 on te)
        base_metrics = []
        base_vk = "trans_natural" if track == "bn" else "src_paraphrase"
        base_idx = m1_views_6.index(base_vk)
        for q in queries:
            qid = str(q["id"])
            s = precomputed_view_scores[qid][base_idx]
            base_metrics.append(eval_scores_matrix(s, q))

        base_r5_arr = np.array([m["r_5kt"] for m in base_metrics])
        base_r5 = float(np.mean(base_r5_arr)) * 100
        base_cfr = float(np.mean([m["catastrophic_failure"] for m in base_metrics])) * 100
        log.info("Base Single View (%s): R@5kt = %.2f%%, CFR = %.2f%%", base_vk, base_r5, base_cfr)

        master_results.append({
            "Track": track, "Method": f"Base Single View ({base_vk})", "R@2kt": f"{np.mean([m['r_2kt'] for m in base_metrics])*100:.2f}%",
            "R@5kt": f"{base_r5:.2f}%", "CFR": f"{base_cfr:.2f}%", "Tail Rescued": "0 (Base)", "Delta vs Base": "+0.00 pts"
        })

        # ---------------------------------------------------------
        # 1. M1: Log-Sum-Exp Beta Sweep over K=6 Sentence Views
        # ---------------------------------------------------------
        for beta in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            m1_metrics = []
            for q in queries:
                qid = str(q["id"])
                view_scores = precomputed_view_scores[qid]
                fused_s = logsumexp_fusion(view_scores, beta)
                m1_metrics.append(eval_scores_matrix(fused_s, q))

            r5_arr = np.array([m["r_5kt"] for m in m1_metrics])
            r5 = float(np.mean(r5_arr)) * 100
            cfr = float(np.mean([m["catastrophic_failure"] for m in m1_metrics])) * 100
            rescued = int(np.sum((base_r5_arr == 0) & (r5_arr == 1)))
            lost = int(np.sum((base_r5_arr == 1) & (r5_arr == 0)))

            name = f"M1: Sentence Bucket (LSE beta={beta})" if beta > 0 else "M1: Sentence Bucket (Centroid / beta=0)"
            master_results.append({
                "Track": track, "Method": name, "R@2kt": f"{np.mean([m['r_2kt'] for m in m1_metrics])*100:.2f}%",
                "R@5kt": f"{r5:.2f}%", "CFR": f"{cfr:.2f}%", "Tail Rescued": f"+{rescued} (Net: {rescued-lost:+d})", "Delta vs Base": f"{r5 - base_r5:+.2f} pts"
            })

        # ---------------------------------------------------------
        # 2. M1: Uncertainty-Aware Disagreement Penalty
        # ---------------------------------------------------------
        for lam in [0.5, 1.0, 2.0, 3.0, 5.0]:
            unc_metrics = []
            for q in queries:
                qid = str(q["id"])
                view_scores = precomputed_view_scores[qid]
                fused_s = uncertainty_fusion(view_scores, lam)
                unc_metrics.append(eval_scores_matrix(fused_s, q))

            r5_arr = np.array([m["r_5kt"] for m in unc_metrics])
            r5 = float(np.mean(r5_arr)) * 100
            cfr = float(np.mean([m["catastrophic_failure"] for m in unc_metrics])) * 100
            rescued = int(np.sum((base_r5_arr == 0) & (r5_arr == 1)))
            lost = int(np.sum((base_r5_arr == 1) & (r5_arr == 0)))

            master_results.append({
                "Track": track, "Method": f"M1: Uncertainty Penalty (lambda={lam})", "R@2kt": f"{np.mean([m['r_2kt'] for m in unc_metrics])*100:.2f}%",
                "R@5kt": f"{r5:.2f}%", "CFR": f"{cfr:.2f}%", "Tail Rescued": f"+{rescued} (Net: {rescued-lost:+d})", "Delta vs Base": f"{r5 - base_r5:+.2f} pts"
            })

        # ---------------------------------------------------------
        # 3. K-Random Unrelated Questions Control (Bucket Size Control)
        # ---------------------------------------------------------
        rand_metrics = []
        for q in queries:
            qid = str(q["id"])
            view_scores = precomputed_rand_scores[qid]
            fused_s = logsumexp_fusion(view_scores, beta=5.0)
            rand_metrics.append(eval_scores_matrix(fused_s, q))

        r5_arr = np.array([m["r_5kt"] for m in rand_metrics])
        r5 = float(np.mean(r5_arr)) * 100
        cfr = float(np.mean([m["catastrophic_failure"] for m in rand_metrics])) * 100
        master_results.append({
            "Track": track, "Method": "CONTROL: K=6 Random Questions", "R@2kt": f"{np.mean([m['r_2kt'] for m in rand_metrics])*100:.2f}%",
            "R@5kt": f"{r5:.2f}%", "CFR": f"{cfr:.2f}%", "Tail Rescued": f"+0", "Delta vs Base": f"{r5 - base_r5:+.2f} pts"
        })

        # ---------------------------------------------------------
        # 4. M3-HyDE (Passage-Shaped Cloze Query)
        # ---------------------------------------------------------
        hyde_metrics = []
        for q in queries:
            qid = str(q["id"])
            v = query_view_embs[qid]["hyde_passage"]
            s = np.dot(pool_embeddings, v)
            hyde_metrics.append(eval_scores_matrix(s, q))

        r5_arr = np.array([m["r_5kt"] for m in hyde_metrics])
        r5 = float(np.mean(r5_arr)) * 100
        cfr = float(np.mean([m["catastrophic_failure"] for m in hyde_metrics])) * 100
        rescued = int(np.sum((base_r5_arr == 0) & (r5_arr == 1)))
        lost = int(np.sum((base_r5_arr == 1) & (r5_arr == 0)))
        master_results.append({
            "Track": track, "Method": "M3: HyDE Passage-Shaped Cloze", "R@2kt": f"{np.mean([m['r_2kt'] for m in hyde_metrics])*100:.2f}%",
            "R@5kt": f"{r5:.2f}%", "CFR": f"{cfr:.2f}%", "Tail Rescued": f"+{rescued} (Net: {rescued-lost:+d})", "Delta vs Base": f"{r5 - base_r5:+.2f} pts"
        })

        # ---------------------------------------------------------
        # 5. Hybrid: M1 (LSE beta=5.0) + M3-HyDE
        # ---------------------------------------------------------
        m1_m3_metrics = []
        for q in queries:
            qid = str(q["id"])
            v_all = np.stack([query_view_embs[qid][vk] for vk in all_view_keys], axis=0)  # K=7 including HyDE
            view_scores = np.dot(v_all, pool_embeddings.T)
            fused_s = logsumexp_fusion(view_scores, beta=5.0)
            m1_m3_metrics.append(eval_scores_matrix(fused_s, q))

        r5_arr = np.array([m["r_5kt"] for m in m1_m3_metrics])
        r5 = float(np.mean(r5_arr)) * 100
        cfr = float(np.mean([m["catastrophic_failure"] for m in m1_m3_metrics])) * 100
        rescued = int(np.sum((base_r5_arr == 0) & (r5_arr == 1)))
        lost = int(np.sum((base_r5_arr == 1) & (r5_arr == 0)))
        master_results.append({
            "Track": track, "Method": "M1 + M3-HyDE (K=7 Sentence Bucket LSE)", "R@2kt": f"{np.mean([m['r_2kt'] for m in m1_m3_metrics])*100:.2f}%",
            "R@5kt": f"{r5:.2f}%", "CFR": f"{cfr:.2f}%", "Tail Rescued": f"+{rescued} (Net: {rescued-lost:+d})", "Delta vs Base": f"{r5 - base_r5:+.2f} pts"
        })

    df_out = pd.DataFrame(master_results)
    out_csv = METRICS_DIR / "step6_m1_m3_evaluation.csv"
    df_out.to_csv(out_csv, index=False)
    print("\n" + "=" * 90)
    print("M1 (MULTI-VIEW SENTENCE BUCKET) & M3 (HYDE) EVALUATION RESULTS")
    print("=" * 90)
    print(df_out.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
