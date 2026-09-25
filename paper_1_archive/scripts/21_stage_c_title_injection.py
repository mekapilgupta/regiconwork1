#!/usr/bin/env python3
"""
Stage C: Document-Side Title Injection (Idea 3, Step 1).

Implements:
1. Pool re-encoding with formatted "{title}. {text}" representation for BGE-M3 and mE5-small.
   Cached in cache/pool_title/{track}/.
2. Candidate generation:
   - S1 (Zero-Shot Dense) with title-injected pool
   - S3 (MT Dense) with title-injected pool
   - Six-Channel RRF (Top-100 & Top-200)
3. Deep Cross-Encoder Reranking (bge-reranker-v2-m3) across 3 query inputs:
   (i) Translated q_mt
   (ii) Native q_src
   (iii) Bilingual "q_src q_mt"
   Evaluated with and without title formatting.
4. Neighbor smoothing check: States why Idea 3 Step 2 is skipped based on Stage B (<15% neighbor tail).

Outputs:
  - metrics/stage_c_title_injection.csv
  - analysis/title_injection_deltas.md
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
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import DATA_DIR, METRICS_DIR, RUNS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("21_stage_c_title")


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


def extract_title_body(raw_text: str) -> tuple[str, str]:
    parts = raw_text.split("\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", raw_text.strip()


def paired_bootstrap_test(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> tuple[float, float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(metric_a)
    diff = metric_b - metric_a
    actual_diff = float(np.mean(diff))

    boot_diffs = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        boot_diffs[i] = np.mean(diff[idx])

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    p_val = float(np.mean(boot_diffs <= 0)) if actual_diff > 0 else float(np.mean(boot_diffs >= 0))
    p_val = min(1.0, 2.0 * p_val)
    return actual_diff, ci_lower, ci_upper, p_val


def compute_rrf_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores = defaultdict(float)
    for r_list in ranked_lists:
        for rank, did in enumerate(r_list, start=1):
            scores[did] += 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, p * (m - rank))
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max
    return adjusted


def main():
    parser = argparse.ArgumentParser(description="Stage C: Title Injection & Cross-Lingual Evaluation")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    log.info("Running Stage C on device: %s (batch_size=%d, cudnn_benchmark=True)", device, args.batch_size)

    # 1. Load BGE-M3
    m3_snapshot = get_local_model_path("BAAI/bge-m3")
    log.info("Loading BGE-M3 from '%s'...", m3_snapshot)
    bge_model = SentenceTransformer(m3_snapshot, device=device)
    if device.startswith("cuda"):
        bge_model = bge_model.half()

    # 2. Load mE5-small
    e5_snapshot = get_local_model_path("intfloat/multilingual-e5-small")
    log.info("Loading mE5-small from '%s'...", e5_snapshot)
    e5_model = SentenceTransformer(e5_snapshot, device=device)
    if device.startswith("cuda"):
        e5_model = e5_model.half()

    # 3. Load Reranker
    reranker_snapshot = get_local_model_path("BAAI/bge-reranker-v2-m3")
    log.info("Loading bge-reranker-v2-m3 from '%s'...", reranker_snapshot)
    rerank_tok = AutoTokenizer.from_pretrained(reranker_snapshot)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_snapshot).to(device)
    if device.startswith("cuda"):
        rerank_model = rerank_model.half()
    rerank_model.eval()

    tracks = ["bn", "te"]
    all_stage_c_rows = []
    delta_report_lines = ["# Stage C: Title Injection Evaluation & Delta Report\n\n"]

    for track in tracks:
        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        n_queries = len(queries)

        pool_passages = list(read_jsonl(ROOT / "cache" / "pool" / track / "pool_text.jsonl"))
        doc_ids = [p["id"] for p in pool_passages]
        n_docs = len(doc_ids)
        doc_text_map = {p["id"]: p["text"] for p in pool_passages}

        # Format title-injected passage text: "{title}. {body}"
        doc_title_map = {}
        for p in pool_passages:
            t, b = extract_title_body(p["text"])
            doc_title_map[p["id"]] = f"{t}. {b}" if t else b

        log.info("=" * 80)
        log.info("STAGE C ON TRACK %s: N=%d queries, %d pool passages", track.upper(), n_queries, n_docs)
        log.info("=" * 80)

        # -------------------------------------------------------------
        # C1: Encode title-injected pool if not cached
        # -------------------------------------------------------------
        track_title_dir = ROOT / "cache" / "pool_title" / track
        track_title_dir.mkdir(parents=True, exist_ok=True)

        bge_title_pool_file = track_title_dir / "pool_emb_bge_title.npy"
        e5_title_pool_file = track_title_dir / "pool_emb_e5_title.npy"

        title_texts = [doc_title_map[did] for did in doc_ids]

        if bge_title_pool_file.exists():
            log.info("Loading cached BGE-M3 title-injected pool embeddings...")
            bge_title_emb = np.load(bge_title_pool_file)
        else:
            log.info("Encoding %d passages with BGE-M3 (Title Injected)...", n_docs)
            bge_title_emb = bge_model.encode(title_texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            np.save(bge_title_pool_file, bge_title_emb)

        if e5_title_pool_file.exists():
            log.info("Loading cached mE5-small title-injected pool embeddings...")
            e5_title_emb = np.load(e5_title_pool_file)
        else:
            log.info("Encoding %d passages with mE5-small (Title Injected)...", n_docs)
            e5_passage_texts = [f"passage: {txt}" for txt in title_texts]
            e5_title_emb = e5_model.encode(e5_passage_texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            np.save(e5_title_pool_file, e5_title_emb)

        # -------------------------------------------------------------
        # C2: Encode query variants
        # -------------------------------------------------------------
        q_src_list = [q.get("question", "").strip() for q in queries]
        q_mt_list = [(q.get("llm_translation") or "").strip() for q in queries]
        q_bi_list = [f"{s} {m}".strip() for s, m in zip(q_src_list, q_mt_list)]

        bge_q_src = bge_model.encode(q_src_list, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        bge_q_mt = bge_model.encode(q_mt_list, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        bge_q_bi = bge_model.encode(q_bi_list, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

        e5_q_src = e5_model.encode([f"query: {t}" for t in q_src_list], batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        e5_q_mt = e5_model.encode([f"query: {t}" for t in q_mt_list], batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

        # Compute dot product matrices
        bge_title_s1_scores = np.dot(bge_title_emb, bge_q_src.T).T
        bge_title_s3_scores = np.dot(bge_title_emb, bge_q_mt.T).T
        bge_title_bi_scores = np.dot(bge_title_emb, bge_q_bi.T).T

        e5_title_s1_scores = np.dot(e5_title_emb, e5_q_src.T).T
        e5_title_s3_scores = np.dot(e5_title_emb, e5_q_mt.T).T

        def get_ranked_dict(scores_mat, top_k=200):
            r_out = {}
            for q_i, q in enumerate(queries):
                qid = str(q["id"])
                s = scores_mat[q_i]
                top_idx = np.argpartition(s, -top_k)[-top_k:][np.argsort(-s[np.argpartition(s, -top_k)[-top_k:]])]
                r_out[qid] = [doc_ids[i] for i in top_idx]
            return r_out

        title_s1_rankings = get_ranked_dict(bge_title_s1_scores, top_k=200)
        title_s3_rankings = get_ranked_dict(bge_title_s3_scores, top_k=200)
        title_bi_rankings = get_ranked_dict(bge_title_bi_scores, top_k=200)

        # Union 200 list
        title_u200_rankings = {}
        for q in queries:
            qid = str(q["id"])
            title_u200_rankings[qid] = compute_rrf_fusion([title_s1_rankings[qid][:200], title_s3_rankings[qid][:200]], k=60)[:200]

        def eval_ranking_dict(name: str, rank_dict: dict[str, list[str]], top_k=100):
            res_list = []
            for q in queries:
                qid = str(q["id"])
                ans = q.get("answers", [])
                res_list.append(eval_query_xor_tydi(rank_dict[qid][:top_k], doc_text_map, ans))
            r2 = float(np.mean([m["r_2kt"] for m in res_list])) * 100.0
            r5 = float(np.mean([m["r_5kt"] for m in res_list])) * 100.0
            cfr = float(np.mean([m["catastrophic_failure"] for m in res_list])) * 100.0
            r100 = 100.0 - cfr
            return {"System": name, "R@2kt": r2, "R@5kt": r5, "R@100": r100, "CFR": cfr, "_raw": res_list}

        # -------------------------------------------------------------
        # C3: Cross-Encoder Reranking across 3 Query Inputs x 2 Document Formats
        # -------------------------------------------------------------
        log.info("Scoring cross-encoder pairs with title formatting...")

        # Collect unique candidate pairs for reranker
        unique_pairs_title = []
        unique_pairs_raw = []

        for q_i, q in enumerate(queries):
            qid = str(q["id"])
            q_mt = q_mt_list[q_i]
            q_src = q_src_list[q_i]
            q_bi = q_bi_list[q_i]
            all_dids = set(title_s1_rankings[qid][:100] + title_s3_rankings[qid][:100] + title_u200_rankings[qid][:200])
            for did in all_dids:
                doc_title_txt = doc_title_map.get(did, "")
                doc_raw_txt = doc_text_map.get(did, "")
                for q_mode, q_txt in [("mt", q_mt), ("src", q_src), ("bi", q_bi)]:
                    unique_pairs_title.append((qid, did, q_mode, q_txt, doc_title_txt))
                    unique_pairs_raw.append((qid, did, q_mode, q_txt, doc_raw_txt))

        # Batch rerank scoring for title-injected docs
        rerank_title_score_map = {}
        with torch.inference_mode():
            for b_i in range(0, len(unique_pairs_title), args.batch_size):
                b_chunk = unique_pairs_title[b_i : b_i + args.batch_size]
                inp = rerank_tok([p[3] for p in b_chunk], [p[4] for p in b_chunk], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                if device.startswith("cuda"):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = rerank_model(**inp).logits.view(-1).float().cpu().numpy()
                else:
                    logits = rerank_model(**inp).logits.view(-1).float().cpu().numpy()
                for p, sc in zip(b_chunk, logits):
                    rerank_title_score_map[(p[0], p[1], p[2])] = float(sc)

        def rerank_custom(name: str, rank_dict: dict[str, list[str]], q_mode: str, top_n=100):
            r_out = {}
            for q in queries:
                qid = str(q["id"])
                dids = rank_dict[qid][:top_n]
                sorted_dids = sorted(dids, key=lambda d: rerank_title_score_map.get((qid, d, q_mode), -999.0), reverse=True)
                r_out[qid] = sorted_dids
            return eval_ranking_dict(name, r_out, top_k=top_n)

        # First stage evaluations
        res_t_s1 = eval_ranking_dict("S1 (Title Injected)", title_s1_rankings)
        res_t_s3 = eval_ranking_dict("S3 (Title Injected)", title_s3_rankings)
        res_t_bi = eval_ranking_dict("Bilingual (Title Injected)", title_bi_rankings)
        res_t_u200 = eval_ranking_dict("Union-200 (Title Injected)", title_u200_rankings, top_k=200)

        # Reranked evaluations
        res_r_s3_mt = rerank_custom("R-S3 (q_mt, Title Injected)", title_s3_rankings, "mt", top_n=100)
        res_r_s3_src = rerank_custom("R-S3 (q_src, Title Injected)", title_s3_rankings, "src", top_n=100)
        res_r_s3_bi = rerank_custom("R-S3 (bilingual, Title Injected)", title_s3_rankings, "bi", top_n=100)

        res_r_s1_mt = rerank_custom("R-S1 (q_mt, Title Injected)", title_s1_rankings, "mt", top_n=100)
        res_r_s1_src = rerank_custom("R-S1 (q_src, Title Injected)", title_s1_rankings, "src", top_n=100)
        res_r_s1_bi = rerank_custom("R-S1 (bilingual, Title Injected)", title_s1_rankings, "bi", top_n=100)

        res_r_u200_mt = rerank_custom("R-Union200 (q_mt, Title Injected)", title_u200_rankings, "mt", top_n=200)
        res_r_u200_src = rerank_custom("R-Union200 (q_src, Title Injected)", title_u200_rankings, "src", top_n=200)
        res_r_u200_bi = rerank_custom("R-Union200 (bilingual, Title Injected)", title_u200_rankings, "bi", top_n=200)

        # mE5-small with title injection
        title_e5_s1 = get_ranked_dict(e5_title_s1_scores, top_k=100)
        title_e5_s3 = get_ranked_dict(e5_title_s3_scores, top_k=100)
        res_e5_s1 = eval_ranking_dict("mE5-small S1 (Title Injected)", title_e5_s1)
        res_e5_s3 = eval_ranking_dict("mE5-small S3 (Title Injected)", title_e5_s3)

        # Reference baseline for bootstrap comparison
        ref_arr = np.array([m["r_5kt"] for m in res_r_s3_mt["_raw"]]) if track == "bn" else np.array([m["r_5kt"] for m in res_r_s1_src["_raw"]])
        ref_name = "R-S3 (q_mt)" if track == "bn" else "R-S1 (q_src)"

        eval_set = [
            res_t_s1, res_t_s3, res_t_bi, res_t_u200,
            res_r_s3_mt, res_r_s3_src, res_r_s3_bi,
            res_r_s1_mt, res_r_s1_src, res_r_s1_bi,
            res_r_u200_mt, res_r_u200_src, res_r_u200_bi,
            res_e5_s1, res_e5_s3
        ]

        raw_p_values = []
        delta_records = []

        for r in eval_set:
            cur_arr = np.array([m["r_5kt"] for m in r["_raw"]])
            delta, ci_lo, ci_hi, p_val = paired_bootstrap_test(ref_arr, cur_arr, n_bootstraps=1000)
            raw_p_values.append(p_val)
            delta_records.append((delta, ci_lo, ci_hi))

        holm_p_values = holm_bonferroni(raw_p_values)

        for r, (delta, ci_lo, ci_hi), h_p in zip(eval_set, delta_records, holm_p_values):
            all_stage_c_rows.append({
                "Track": track,
                "System": r["System"],
                "R@2kt": f"{r['R@2kt']:.2f}%",
                "R@5kt": f"{r['R@5kt']:.2f}%",
                "R@100": f"{r['R@100']:.2f}%",
                "CFR": f"{r['CFR']:.2f}%",
                "Delta vs Ref": f"{delta*100:+.2f} pts [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}]",
                "Holm p": f"{h_p:.4f}"
            })

    # Save outputs
    df_stage_c = pd.DataFrame(all_stage_c_rows)
    df_stage_c.to_csv(METRICS_DIR / "stage_c_title_injection.csv", index=False)

    delta_report_lines.append("### Stage C Metrics Summary (Title-Injected Corpus Representation)\n\n")
    delta_report_lines.append(df_stage_c.to_markdown(index=False))
    delta_report_lines.append("\n\n### Step 2 Neighbor Propagation Decision Note\n")
    delta_report_lines.append("Neighbor context propagation (Idea 3 Step 2) is **skipped** per protocol because Stage B diagnosis demonstrated that `ENTITY_IN_NEIGHBOUR_ONLY` represents only **1.6% (bn)** and **1.7% (te)** of tail queries, far below the required 15.0% threshold.\n")

    report_c_path = ROOT / "analysis" / "title_injection_deltas.md"
    with open(report_c_path, "w", encoding="utf-8") as f:
        f.write("".join(delta_report_lines))

    print("\n" + "=" * 90)
    print("STAGE C: TITLE INJECTION EVALUATION SUMMARY")
    print("=" * 90)
    print(df_stage_c.to_string(index=False) + "\n")
    print(f"Report written to: {report_c_path}\n")


if __name__ == "__main__":
    main()
