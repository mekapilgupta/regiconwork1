#!/usr/bin/env python3
"""
Stage D: QPP Evaluation Done Properly (AUROC, AURC, and Risk-Coverage Curves).

Implements:
1. Signal evaluation:
   (a) a1, a2, a3 cross-channel overlap features
   (b) NQC / score standard deviation of top-100 dense scores
   (c) Reranker score margin (top1 - top2)
   (d) Reranker top-1 score alone
   (e) Out-of-fold Logistic Regression on (a)-(c) across 5 folds
2. Performance metrics:
   - AUROC (ranking confidence vs binary success R@5kt)
   - AURC (Area under the Risk-Coverage Curve)
   - Risk-Coverage curve at 10% steps (10% to 100%) compared against:
     * RANDOM baseline (flat error rate)
     * ORACLE baseline (all errors deferred to the tail)
3. Direct statistical test: Does Logistic Regression (e) beat Reranker Score alone (d)?
4. Plots:
   - plots/risk_coverage_bn.png
   - plots/risk_coverage_te.png

Outputs:
  - metrics/qpp_auroc_aurc.csv
  - metrics/stage_d_risk_coverage_table.csv
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.config import DATA_DIR, METRICS_DIR, RUNS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("22_stage_d_qpp")


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


def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """Computes Area Under the Risk-Coverage curve using trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(risks, coverages))
    return float(np.trapz(risks, coverages))


def main():
    parser = argparse.ArgumentParser(description="Stage D: QPP Evaluation and Risk-Coverage Curves")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    log.info("Running Stage D on device: %s (batch_size=%d, cudnn_benchmark=True)", device, args.batch_size)

    # 1. Load BGE-M3
    m3_snapshot = get_local_model_path("BAAI/bge-m3")
    log.info("Loading BGE-M3 from '%s'...", m3_snapshot)
    tokenizer = AutoTokenizer.from_pretrained(m3_snapshot)
    base_model = AutoModel.from_pretrained(m3_snapshot).to(device)
    if device.startswith("cuda"):
        base_model = base_model.half()
    base_model.eval()

    # Load sparse and colbert heads
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
    qpp_summary_records = []
    all_rc_curve_records = []

    for track in tracks:
        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        n_queries = len(queries)
        doc_text_map = load_doc_text_map(track)

        pool_passages = list(read_jsonl(ROOT / "cache" / "pool" / track / "pool_text.jsonl"))
        doc_ids = [p["id"] for p in pool_passages]
        n_docs = len(doc_ids)

        log.info("=" * 80)
        log.info("STAGE D ON TRACK %s: N=%d queries", track.upper(), n_queries)
        log.info("=" * 80)

        inverted_index = defaultdict(list)
        for d_i, p in enumerate(pool_passages):
            words = set(re.findall(r"\w+", p["text"].lower()))
            for w in words:
                inverted_index[w].append(d_i)

        s2_trec_file = RUNS_DIR / track / f"S2__{track}.trec"
        if not s2_trec_file.exists():
            s2_trec_file = RUNS_DIR / f"S2__{track}.trec"
        s2_run_map = defaultdict(list)
        if s2_trec_file.exists():
            with open(s2_trec_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        s2_run_map[parts[0]].append(parts[2])

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

        def get_sparse_ranked_list(q_tokens, q_weights, top_k=100):
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

        # Extract features per query
        features_a1 = []
        features_a2 = []
        features_a3 = []
        features_nqc = []
        features_rerank_margin = []
        features_rerank_top1 = []

        base_scores = dense_mt_scores if track == "bn" else dense_src_scores
        candidates_dict = {}

        # Collect unique pairs for reranker scoring
        pairs_to_score = []
        for q_i, q in enumerate(queries):
            qid = str(q["id"])
            q_txt = (q.get("llm_translation") or q.get("question") or "").strip()
            s_base = base_scores[q_i]
            top_idx = np.argpartition(s_base, -100)[-100:][np.argsort(-s_base[np.argpartition(s_base, -100)[-100:]])]
            dids = [doc_ids[i] for i in top_idx]
            candidates_dict[qid] = dids
            for did in dids:
                pairs_to_score.append((qid, did, q_txt, doc_text_map.get(did, "")))

        log.info("Scoring cross-encoder candidates for QPP on '%s'...", track)
        rerank_score_map = {}
        with torch.inference_mode():
            for b_i in range(0, len(pairs_to_score), args.batch_size):
                b_chunk = pairs_to_score[b_i : b_i + args.batch_size]
                inp = rerank_tok([p[2] for p in b_chunk], [p[3] for p in b_chunk], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                if device.startswith("cuda"):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = rerank_model(**inp).logits.view(-1).float().cpu().numpy()
                else:
                    logits = rerank_model(**inp).logits.view(-1).float().cpu().numpy()
                for p, sc in zip(b_chunk, logits):
                    rerank_score_map[(p[0], p[1])] = float(sc)

        binary_labels = []

        for q_i, q in enumerate(queries):
            qid = str(q["id"])
            ans = q.get("answers", [])
            dids = candidates_dict[qid]

            # Reranked candidates
            sorted_dids = sorted(dids, key=lambda d: rerank_score_map.get((qid, d), -999.0), reverse=True)
            top_scores = [rerank_score_map.get((qid, d), -999.0) for d in sorted_dids]
            
            top1_sc = top_scores[0]
            top2_sc = top_scores[1] if len(top_scores) > 1 else top_scores[0]
            margin = float(top1_sc - top2_sc)

            eval_res = eval_query_xor_tydi(sorted_dids, doc_text_map, ans)
            binary_labels.append(int(eval_res["r_5kt"] > 0))

            # Features
            top_src_10 = [doc_ids[i] for i in np.argpartition(dense_src_scores[q_i], -10)[-10:]]
            top_mt_10 = [doc_ids[i] for i in np.argpartition(dense_mt_scores[q_i], -10)[-10:]]
            top_sp_10 = s2_run_map[qid][:10] if (qid in s2_run_map and len(s2_run_map[qid]) > 0) else get_sparse_ranked_list(ids_mt[q_i], sp_mt[q_i], top_k=10)

            a1 = len(set(top_src_10).intersection(set(top_mt_10))) / 10.0
            a2 = len(set(top_mt_10).intersection(set(top_sp_10))) / 10.0
            a3 = (a1 + a2) / 2.0

            top100_s = np.partition(base_scores[q_i], -100)[-100:]
            nqc = float(np.std(top100_s) / (np.mean(top100_s) + 1e-6))

            features_a1.append(a1)
            features_a2.append(a2)
            features_a3.append(a3)
            features_nqc.append(nqc)
            features_rerank_margin.append(margin)
            features_rerank_top1.append(float(top1_sc))

        binary_labels = np.array(binary_labels)
        f_a1 = np.array(features_a1)
        f_a2 = np.array(features_a2)
        f_a3 = np.array(features_a3)
        f_nqc = np.array(features_nqc)
        f_margin = np.array(features_rerank_margin)
        f_top1 = np.array(features_rerank_top1)

        # 5-Fold OOF Logistic Regression on features (a)-(c)
        X_features = np.stack([f_a1, f_a2, f_a3, f_nqc, f_margin], axis=1)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(n_queries)

        for train_idx, val_idx in kf.split(X_features):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_features[train_idx], binary_labels[train_idx])
            oof_probs[val_idx] = clf.predict_proba(X_features[val_idx])[:, 1]

        # -------------------------------------------------------------
        # Compute AUROC & AURC for all confidence predictors
        # -------------------------------------------------------------
        predictors = {
            "(a) Cross-channel Overlap (a3)": f_a3,
            "(b) Dense Score NQC": f_nqc,
            "(c) Reranker Margin (top1 - top2)": f_margin,
            "(d) Reranker Top-1 Score": f_top1,
            "(e) Logistic Regression OOF (a-c)": oof_probs
        }

        # Step coverages: 10% to 100%
        coverage_steps = np.linspace(0.1, 1.0, 10)

        # Compute curves
        curves_dict = {}
        for p_name, p_scores in predictors.items():
            auroc = roc_auc_score(binary_labels, p_scores)
            
            # Risk-coverage
            sorted_idx = np.argsort(-p_scores)
            risks = []
            for cov in coverage_steps:
                k_cov = max(1, int(round(cov * n_queries)))
                cov_idx = sorted_idx[:k_cov]
                cov_acc = np.mean(binary_labels[cov_idx])
                cov_risk = 1.0 - cov_acc
                risks.append(cov_risk)
            risks = np.array(risks)
            aurc = compute_aurc(coverage_steps, risks)

            curves_dict[p_name] = risks
            qpp_summary_records.append({
                "Track": track,
                "Predictor": p_name,
                "AUROC": f"{auroc:.4f}",
                "AURC": f"{aurc:.4f}",
                "Risk @ 50% Coverage": f"{risks[4]*100:.2f}%",
                "Risk @ 100% Coverage": f"{risks[-1]*100:.2f}%"
            })

        # Random & Oracle Baselines
        base_error = 1.0 - np.mean(binary_labels)
        random_risks = np.full_like(coverage_steps, base_error)
        curves_dict["Random Baseline (Flat)"] = random_risks

        # Oracle Curve (all errors postponed to tail)
        oracle_risks = []
        n_pos = int(np.sum(binary_labels))
        for cov in coverage_steps:
            k_cov = max(1, int(round(cov * n_queries)))
            if k_cov <= n_pos:
                oracle_risks.append(0.0)
            else:
                n_err = k_cov - n_pos
                oracle_risks.append(n_err / k_cov)
        curves_dict["Oracle Baseline (Perfect)"] = np.array(oracle_risks)

        # Record tabular risk-coverage
        for step_i, cov in enumerate(coverage_steps):
            all_rc_curve_records.append({
                "Track": track,
                "Coverage": f"{cov*100:.0f}%",
                "Reranker Top-1 Risk": f"{curves_dict['(d) Reranker Top-1 Score'][step_i]*100:.2f}%",
                "LogReg OOF Risk": f"{curves_dict['(e) Logistic Regression OOF (a-c)'][step_i]*100:.2f}%",
                "Random Risk": f"{random_risks[step_i]*100:.2f}%",
                "Oracle Risk": f"{curves_dict['Oracle Baseline (Perfect)'][step_i]*100:.2f}%"
            })

        # -------------------------------------------------------------
        # Generate Plot: plots/risk_coverage_{track}.png
        # -------------------------------------------------------------
        plt.figure(figsize=(8, 5.5), dpi=300)
        plt.plot(coverage_steps * 100, curves_dict["(d) Reranker Top-1 Score"] * 100, label="(d) Reranker Top-1 Score", color="#1f77b4", lw=2, marker="o")
        plt.plot(coverage_steps * 100, curves_dict["(e) Logistic Regression OOF (a-c)"] * 100, label="(e) Logistic Regression OOF (a-c)", color="#2ca02c", lw=2, marker="s")
        plt.plot(coverage_steps * 100, random_risks * 100, label="Random Baseline (Flat)", color="#d62728", lw=1.5, ls="--")
        plt.plot(coverage_steps * 100, curves_dict["Oracle Baseline (Perfect)"] * 100, label="Oracle (Perfect Ordering)", color="#7f7f7f", lw=1.5, ls=":")
        
        plt.title(f"Risk-Coverage Curve — Track {track.upper()} (N={n_queries})", fontsize=13, pad=12)
        plt.xlabel("Coverage (% of Queries Evaluated)", fontsize=11)
        plt.ylabel("Risk / Error Rate (% Gold Not in Top-5kt)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left", frameon=True)
        plt.tight_layout()

        plot_file = ROOT / "plots" / f"risk_coverage_{track}.png"
        plt.savefig(plot_file)
        plt.close()
        log.info("Saved risk-coverage plot to %s", plot_file)

    df_qpp = pd.DataFrame(qpp_summary_records)
    df_qpp.to_csv(METRICS_DIR / "qpp_auroc_aurc.csv", index=False)

    df_rc_tab = pd.DataFrame(all_rc_curve_records)
    df_rc_tab.to_csv(METRICS_DIR / "stage_d_risk_coverage_table.csv", index=False)

    print("\n" + "=" * 90)
    print("STAGE D: QPP AUROC & AURC PERFORMANCE SUMMARY")
    print("=" * 90)
    print(df_qpp.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
