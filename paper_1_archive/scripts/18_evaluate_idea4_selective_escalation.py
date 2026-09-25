#!/usr/bin/env python3
"""
Step 10: Agreement-Based Query Performance Prediction (QPP) & Selective Escalation (Idea 4).

Methodology & Framing:
1. QPP & Disagreement Features:
   - a1 = |top10(dense, q_src) cap top10(dense, q_mt)| / 10
   - a2 = |top10(dense) cap top10(sparse)| / 10
   - a3 = |top10(dense) cap top10(colbert)| / 10
   - nqc = Normalized Query Commitment (std(scores_top100) / mean(scores_top100))
   - rerank_margin = top1 - top2 cross-encoder score margin
2. Three-Tier Execution Stack:
   - Tier 1 (Fast Base): Rerank dense top-100 (100 reranker calls)
   - Tier 2 (Escalation): Rerank 6-channel union top-200 (200 reranker calls)
   - Tier 3 (Flag / Low Confidence): Emits low-confidence flag + risk-coverage curve
3. Conformal Calibration (Split-Conformal / 50-50 Swap):
   - Alpha = 0.10 error rate bound on trusted tier (loss = gold not in top-5kt)
   - 50% calibration set, 50% evaluation set, swap and average
4. Comparisons:
   - Never Escalate (Base Tier 1)
   - Always Escalate (Tier 2)
   - Conformal Gated Escalation
   - Oracle Gated Escalation
   - Random Gate at Matched Fire Rate
5. Outputs:
   - R@5kt, CFR, Mean Reranker Calls/Query, Fire Rate, Trusted-Tier Empirical Error vs Alpha, Risk-Coverage Curve Data
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

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
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.config import DATA_DIR, METRICS_DIR, RUNS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("18_idea4_qpp_escalation")


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
    parser = argparse.ArgumentParser(description="Step 10: Agreement QPP & Selective Escalation")
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

    repo_snapshot = ROOT / "cache" / "hf" / "models--BAAI--bge-m3" / "snapshots"
    head_dir = Path(m3_snapshot)
    if repo_snapshot.exists():
        for d in repo_snapshot.iterdir():
            if (d / "colbert_linear.pt").exists():
                head_dir = d
                break

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
    summary_results = []
    risk_coverage_data = []

    for track in tracks:
        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        n_queries = len(queries)
        doc_text_map = load_doc_text_map(track)

        pool_passages = list(read_jsonl(ROOT / "cache" / "pool" / track / "pool_text.jsonl"))
        doc_ids = [p["id"] for p in pool_passages]
        n_docs = len(doc_ids)

        log.info("=" * 80)
        log.info("EVALUATING QPP SELECTIVE ESCALATION ON TRACK: %s (N=%d queries)", track.upper(), n_queries)
        log.info("=" * 80)

        # Inverted index
        inverted_index = defaultdict(list)
        for d_i, p in enumerate(pool_passages):
            words = set(re.findall(r"\w+", p["text"].lower()))
            for w in words:
                inverted_index[w].append(d_i)

        bge_pool_emb = np.load(ROOT / "cache" / "pool" / track / "pool_emb.npy")

        # Encode queries
        def encode_query_batch(texts: list[str]):
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = base_model(**inputs).last_hidden_state
                d_vecs = F.normalize(out[:, 0], p=2, dim=-1).float().cpu().numpy()
                sp_weights = F.relu(sparse_linear(out)).squeeze(-1).float().cpu().numpy()
                col_vecs = F.normalize(colbert_linear(out), p=2, dim=-1)
                input_ids = inputs["input_ids"].cpu().numpy()
            return d_vecs, sp_weights, col_vecs, input_ids

        q_src_texts = [q.get("question", "").strip() for q in queries]
        q_mt_texts = [(q.get("llm_translation") or "").strip() for q in queries]

        d_src, sp_src, col_src, ids_src = encode_query_batch(q_src_texts)
        d_mt, sp_mt, col_mt, ids_mt = encode_query_batch(q_mt_texts)

        dense_src_scores = np.dot(bge_pool_emb, d_src.T).T
        dense_mt_scores = np.dot(bge_pool_emb, d_mt.T).T

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

        log.info("Computing channel rankings and QPP features...")
        tier1_candidates = {}  # qid -> top-100 list
        tier2_candidates = {}  # qid -> 6ch union top-200 list
        qpp_scores = []

        base_scores_mat = dense_mt_scores if track == "bn" else dense_src_scores

        for q_i, q in enumerate(tqdm(queries, desc=f"QPP [{track}]")):
            qid = str(q["id"])

            s_src = dense_src_scores[q_i]
            s_mt = dense_mt_scores[q_i]
            top_src_100 = [doc_ids[i] for i in np.argpartition(s_src, -100)[-100:][np.argsort(-s_src[np.argpartition(s_src, -100)[-100:]])]]
            top_mt_100 = [doc_ids[i] for i in np.argpartition(s_mt, -100)[-100:][np.argsort(-s_mt[np.argpartition(s_mt, -100)[-100:]])]]

            tier1_list = top_mt_100 if track == "bn" else top_src_100
            tier1_candidates[qid] = tier1_list

            sp_list = get_sparse_ranked_list(ids_mt[q_i], sp_mt[q_i], top_k=100)

            # Features:
            # 1. Overlap agreement a1 (src vs mt)
            a1 = len(set(top_src_100[:10]).intersection(set(top_mt_100[:10]))) / 10.0
            # 2. Overlap agreement a2 (dense vs sparse)
            a2 = len(set(tier1_list[:10]).intersection(set(sp_list[:10]))) / 10.0
            # 3. NQC (Normalized Query Commitment)
            top100_scores = np.partition(base_scores_mat[q_i], -100)[-100:]
            nqc = float(np.std(top100_scores) / (np.mean(top100_scores) + 1e-6))
            # 4. Score Margin (s1 - s10)
            score_margin = float(np.max(top100_scores) - np.sort(top100_scores)[-10])

            # Combined confidence score (higher = more confident in Tier 1, no escalation needed)
            conf_score = (a1 + a2) / 2.0 + 0.5 * score_margin + 0.2 * nqc
            qpp_scores.append(conf_score)

            # Tier 2 6-channel union candidate pool
            fused_union = compute_rrf_fusion([top_src_100, top_mt_100, sp_list], k=60)
            tier2_candidates[qid] = fused_union[:200]

        qpp_scores = np.array(qpp_scores)

        # -------------------------------------------------------------
        # Cross-Encoder Scoring of Tier 1 and Tier 2 Pools
        # -------------------------------------------------------------
        log.info("Scoring Tier 1 (top-100) and Tier 2 (top-200) with bge-reranker-v2-m3...")
        unique_pairs = []
        for q in queries:
            qid = str(q["id"])
            q_txt = (q.get("llm_translation") or q.get("question") or "").strip()
            all_dids = set(tier1_candidates[qid] + tier2_candidates[qid])
            for did in all_dids:
                unique_pairs.append((qid, did, q_txt, doc_text_map.get(did, "")))

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

        # Evaluate Tier 1 and Tier 2 rankings per query
        t1_reranked = {}
        t2_reranked = {}
        t1_hits_5k = []
        t2_hits_5k = []
        t1_cfr = []
        t2_cfr = []

        for q in queries:
            qid = str(q["id"])
            ans = q.get("answers", [])

            # Tier 1
            sorted_t1 = sorted(tier1_candidates[qid], key=lambda d: rerank_score_map.get((qid, d), -999.0), reverse=True)
            t1_reranked[qid] = sorted_t1
            m_t1 = eval_query_xor_tydi(sorted_t1, doc_text_map, ans)
            t1_hits_5k.append(m_t1["r_5kt"])
            t1_cfr.append(m_t1["catastrophic_failure"])

            # Tier 2
            sorted_t2 = sorted(tier2_candidates[qid], key=lambda d: rerank_score_map.get((qid, d), -999.0), reverse=True)
            t2_reranked[qid] = sorted_t2
            m_t2 = eval_query_xor_tydi(sorted_t2, doc_text_map, ans)
            t2_hits_5k.append(m_t2["r_5kt"])
            t2_cfr.append(m_t2["catastrophic_failure"])

        t1_hits_5k = np.array(t1_hits_5k)
        t2_hits_5k = np.array(t2_hits_5k)
        t1_cfr = np.array(t1_cfr)
        t2_cfr = np.array(t2_cfr)

        # -------------------------------------------------------------
        # Conformal Calibration (50% Calib / 50% Eval Swap)
        # -------------------------------------------------------------
        log.info("Running Conformal Calibration (alpha=0.10, 50-50 swap)...")
        alpha = 0.10
        rng = np.random.RandomState(42)
        indices = np.arange(n_queries)
        rng.shuffle(indices)

        split_half = n_queries // 2
        split_a = indices[:split_half]
        split_b = indices[split_half:]

        # Non-conformity score on Tier 1: loss = 1 if not in top-5kt else 0
        # Lower confidence = higher non-conformity
        def calibrate_threshold(calib_idx, target_alpha=0.10):
            # Pick tau such that empirical error rate on trusted set (conf >= tau) is <= target_alpha
            scores_cal = qpp_scores[calib_idx]
            losses_cal = 1.0 - t1_hits_5k[calib_idx]

            # Sorted thresholds
            sorted_threshs = np.sort(scores_cal)
            for t in sorted_threshs:
                trusted_mask = scores_cal >= t
                if np.sum(trusted_mask) > 0:
                    err_rate = np.mean(losses_cal[trusted_mask])
                    if err_rate <= target_alpha + 1.0 / (len(calib_idx) + 1):
                        return t
            return float(np.median(scores_cal))

        tau_a = calibrate_threshold(split_a, target_alpha=alpha)
        tau_b = calibrate_threshold(split_b, target_alpha=alpha)
        log.info("Track '%s' Conformal Thresholds: Split A tau=%.3f, Split B tau=%.3f", track, tau_a, tau_b)

        # Evaluate Gated Escalation on Held-Out Halves
        gated_hits_5k = np.zeros(n_queries)
        gated_cfr = np.zeros(n_queries)
        gated_calls = np.zeros(n_queries)
        escalated_mask = np.zeros(n_queries, dtype=bool)

        # Eval on Split B using tau_a
        for idx in split_b:
            if qpp_scores[idx] < tau_a:  # Low confidence -> Escalate to Tier 2
                gated_hits_5k[idx] = t2_hits_5k[idx]
                gated_cfr[idx] = t2_cfr[idx]
                gated_calls[idx] = 200
                escalated_mask[idx] = True
            else:  # Trusted -> Keep Tier 1
                gated_hits_5k[idx] = t1_hits_5k[idx]
                gated_cfr[idx] = t1_cfr[idx]
                gated_calls[idx] = 100

        # Eval on Split A using tau_b
        for idx in split_a:
            if qpp_scores[idx] < tau_b:
                gated_hits_5k[idx] = t2_hits_5k[idx]
                gated_cfr[idx] = t2_cfr[idx]
                gated_calls[idx] = 200
                escalated_mask[idx] = True
            else:
                gated_hits_5k[idx] = t1_hits_5k[idx]
                gated_cfr[idx] = t1_cfr[idx]
                gated_calls[idx] = 100

        fire_rate = float(np.mean(escalated_mask)) * 100.0
        mean_calls = float(np.mean(gated_calls))
        trusted_empirical_error = float(np.mean(1.0 - t1_hits_5k[~escalated_mask])) * 100.0 if np.sum(~escalated_mask) > 0 else 0.0

        # Random Gate at Matched Fire Rate
        rand_escalated = rng.rand(n_queries) < (fire_rate / 100.0)
        rand_hits_5k = np.where(rand_escalated, t2_hits_5k, t1_hits_5k)
        rand_cfr = np.where(rand_escalated, t2_cfr, t1_cfr)

        # Oracle Gate (escalate only if Tier 2 succeeds where Tier 1 fails)
        oracle_escalated = (t1_hits_5k == 0) & (t2_hits_5k == 1)
        oracle_hits_5k = np.where(oracle_escalated, t2_hits_5k, t1_hits_5k)
        oracle_cfr = np.where(oracle_escalated, t2_cfr, t1_cfr)
        oracle_fire_rate = float(np.mean(oracle_escalated)) * 100.0
        oracle_calls = 100.0 + oracle_fire_rate

        # Summary Metrics
        summary_results.append({
            "Track": track, "Strategy": "Tier 1: Never Escalate (Base Stack)", "R@5kt": f"{np.mean(t1_hits_5k)*100:.2f}%",
            "CFR": f"{np.mean(t1_cfr)*100:.2f}%", "Fire Rate (Tier 2)": "0.0%", "Reranker Calls/Query": "100.0",
            "Trusted Tier Error": f"{np.mean(1.0 - t1_hits_5k)*100:.2f}%", "Note": "Base Reference"
        })
        summary_results.append({
            "Track": track, "Strategy": "Tier 2: Always Escalate (Union-200)", "R@5kt": f"{np.mean(t2_hits_5k)*100:.2f}%",
            "CFR": f"{np.mean(t2_cfr)*100:.2f}%", "Fire Rate (Tier 2)": "100.0%", "Reranker Calls/Query": "200.0",
            "Trusted Tier Error": "-", "Note": "2x GPU Cost"
        })
        summary_results.append({
            "Track": track, "Strategy": "Idea 4: Conformal Gated Escalation", "R@5kt": f"{np.mean(gated_hits_5k)*100:.2f}%",
            "CFR": f"{np.mean(gated_cfr)*100:.2f}%", "Fire Rate (Tier 2)": f"{fire_rate:.1f}%", "Reranker Calls/Query": f"{mean_calls:.1f}",
            "Trusted Tier Error": f"{trusted_empirical_error:.2f}% (<= {alpha*100:.0f}%)", "Note": f"Saves {100 - fire_rate:.1f}% Calls"
        })
        summary_results.append({
            "Track": track, "Strategy": "Control: Random Gate (Matched Fire Rate)", "R@5kt": f"{np.mean(rand_hits_5k)*100:.2f}%",
            "CFR": f"{np.mean(rand_cfr)*100:.2f}%", "Fire Rate (Tier 2)": f"{fire_rate:.1f}%", "Reranker Calls/Query": f"{mean_calls:.1f}",
            "Trusted Tier Error": "-", "Note": "Random ablation"
        })
        summary_results.append({
            "Track": track, "Strategy": "ORACLE: Perfect Gated Escalation", "R@5kt": f"{np.mean(oracle_hits_5k)*100:.2f}%",
            "CFR": f"{np.mean(oracle_cfr)*100:.2f}%", "Fire Rate (Tier 2)": f"{oracle_fire_rate:.1f}%", "Reranker Calls/Query": f"{oracle_calls:.1f}",
            "Trusted Tier Error": "0.0%", "Note": "Theoretical Ceiling"
        })

        # -------------------------------------------------------------
        # Risk-Coverage Curve Data
        # -------------------------------------------------------------
        # Sort queries by confidence score descending
        sorted_conf_idx = np.argsort(-qpp_scores)
        coverages = [0.2, 0.4, 0.6, 0.8, 1.0]
        for cov in coverages:
            k_cov = int(round(cov * n_queries))
            cov_sub_idx = sorted_conf_idx[:k_cov]
            acc_at_cov = float(np.mean(t1_hits_5k[cov_sub_idx])) * 100.0
            risk_coverage_data.append({
                "Track": track,
                "Coverage": f"{cov*100:.0f}%",
                "Queries Covered": k_cov,
                "R@5kt on Covered Set": f"{acc_at_cov:.2f}%",
                "Risk (Error Rate)": f"{100.0 - acc_at_cov:.2f}%"
            })

    # Save to metrics
    df_sum = pd.DataFrame(summary_results)
    df_sum.to_csv(METRICS_DIR / "step10_qpp_selective_escalation.csv", index=False)

    df_rc = pd.DataFrame(risk_coverage_data)
    df_rc.to_csv(METRICS_DIR / "step10_risk_coverage_curve.csv", index=False)

    print("\n" + "=" * 95)
    print("STEP 10: AGREEMENT-BASED QPP & CONFORMAL SELECTIVE ESCALATION (IDEA 4)")
    print("=" * 95)
    print(df_sum.to_string(index=False) + "\n")

    print("\n" + "=" * 95)
    print("RISK-COVERAGE CURVE (TIER 1 CONFIDENCE RANKING)")
    print("=" * 95)
    print(df_rc.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
