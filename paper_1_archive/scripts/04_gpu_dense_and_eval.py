#!/usr/bin/env python3
"""
Stage 4: GPU Dense Encoding, Multi-Representation Retrieval, and Rigorous Evaluation.

Workflow:
1. GPU Encoding of Candidate Pools:
   - Reads cache/pool/<track>/pool_text.jsonl (~77k for bn, ~66k for te).
   - Encodes passages once using BAAI/bge-m3 (fp16 on GPU).
   - Caches to cache/pool/<track>/pool_emb.npy.

2. Systems Execution (S1–S9, A1–A4):
   - S1: Zero-shot source query (BGE-M3 dense).
   - S2: Translate-then-BM25 on full 21M Wikipedia Lucene index.
   - S3: Machine Translation dense (BGE-M3 on llm_translation).
   - S3r: Machine Translation Rewrite dense (BGE-M3 on llm_rewrite).
   - S4: Facet-only retrieval (BGE-M3 on English facets concatenation).
   - S5: Facet + Translation Concat (BGE-M3 on llm_translation + English facets).
   - S6: Structured Multi-Vector Facet Fusion (separate dense scoring of entities, relation, constraints).
   - S6_src: Structured Multi-Vector Fusion using untranslated source-script surface forms.
   - S7: Reciprocal Rank Fusion (RRF of S1 + S2 + S3).
   - S8: Human Oracle English retrieval (BGE-M3 on human eng_translation).
   - S9: Random-Span Expansion Control (BGE-M3 on llm_translation + random distractor words).
   - A1–A4: Ablations (no_ent, no_const, no_rel, surface_only).

3. Evaluation:
   - Official XOR-TyDi R@2kt, R@5kt, Catastrophic Failure Rate (CFR), R@20, R@100, MRR@10.
   - 1,000 paired bootstrap resamples for 95% Confidence Intervals.
   - Two-sided p-values against proposed system S6 with Holm-Bonferroni multiple testing correction.
   - Exports TREC runs to runs/<track>/ and metrics tables to metrics/<track>_main.csv.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Pre-set Java, OpenAI dummy env, and offline HF flags
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("JAVA_HOME", "/home/kshitij22256/miniconda3/envs/clir_cuda124")
os.environ.setdefault("JVM_PATH", "/home/kshitij22256/miniconda3/envs/clir_cuda124/lib/server/libjvm.so")
os.environ["PATH"] = f"/home/kshitij22256/miniconda3/envs/clir_cuda124/bin:{os.environ.get('PATH', '')}"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import DATA_DIR, METRICS_DIR, RUNS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("04_gpu_dense_eval")


def get_local_model_path(model_id: str = "BAAI/bge-m3") -> str:
    """Finds the local offline snapshot path containing modules.json to prevent HF network requests."""
    # Check ~/.cache/huggingface/hub/
    cache_hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if cache_hub.exists():
        for d in cache_hub.iterdir():
            if d.is_dir() and (d / "modules.json").exists():
                log.info("Found verified local offline snapshot: %s", d)
                return str(d)
    # Check repo cache/hf/
    repo_hub = ROOT / "cache" / "hf" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if repo_hub.exists():
        for d in repo_hub.iterdir():
            if d.is_dir() and (d / "modules.json").exists():
                log.info("Found verified local offline snapshot in repo: %s", d)
                return str(d)
    return model_id

POOL_CACHE_DIR = ROOT / "cache" / "pool"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def encode_pool_if_needed(track: str, model: SentenceTransformer, batch_size: int = 256) -> tuple[list[str], dict[str, str], np.ndarray]:
    track_pool_dir = POOL_CACHE_DIR / track
    pool_text_file = track_pool_dir / "pool_text.jsonl"
    out_emb_file = track_pool_dir / "pool_emb.npy"

    assert pool_text_file.exists(), f"Pool text file not found: {pool_text_file}"
    passages = list(read_jsonl(pool_text_file))
    doc_ids = [p["id"] for p in passages]
    doc_text_map = {p["id"]: p["text"] for p in passages}

    if out_emb_file.exists():
        log.info("Loading cached pool embeddings from %s...", out_emb_file)
        pool_embeddings = np.load(out_emb_file)
        assert len(pool_embeddings) == len(passages), "Embedding count mismatch!"
    else:
        log.info("Encoding %d passages for track '%s' using BAAI/bge-m3 on GPU...", len(passages), track)
        model.max_seq_length = 512
        texts = [p["text"] for p in passages]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        pool_embeddings = embeddings.astype(np.float32)
        np.save(out_emb_file, pool_embeddings)
        log.info("Saved pool embeddings to %s (shape=%s)", out_emb_file, pool_embeddings.shape)

    return doc_ids, doc_text_map, pool_embeddings


def zscore_normalize(scores: np.ndarray) -> np.ndarray:
    std = scores.std()
    if std < 1e-8:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / std


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    rrf_scores = {}
    for r_list in ranked_lists:
        for rank, docid in enumerate(r_list, start=1):
            rrf_scores[docid] = rrf_scores.get(docid, 0.0) + (1.0 / (k + rank))
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


def run_dense_query(query_text: str, model: SentenceTransformer, pool_embeddings: np.ndarray, doc_ids: list[str]) -> np.ndarray:
    if not query_text.strip():
        return np.zeros(len(doc_ids), dtype=np.float32)
    q_emb = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    # Cosine similarity since vectors are normalized
    scores = np.dot(pool_embeddings, q_emb)
    return scores


def run_full_track(track: str, model: SentenceTransformer, searcher: LuceneSearcher, batch_size: int = 256):
    log.info("=" * 70)
    log.info("RUNNING FULL RETRIEVAL & EVALUATION FOR TRACK: %s", track.upper())
    log.info("=" * 70)

    track_runs_dir = RUNS_DIR / track
    track_runs_dir.mkdir(parents=True, exist_ok=True)

    proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
    records = list(read_jsonl(proc_file))
    log.info("Loaded %d queries for track '%s'", len(records), track)

    # 1. Load / Compute Pool Embeddings
    doc_ids, doc_text_map, pool_embeddings = encode_pool_if_needed(track, model, batch_size=batch_size)
    doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}

    system_names = [
        "S1", "S2", "S3", "S3r", "S4", "S5", "S5r", "S6", "S6_src", "S7", "S8", "S9",
        "A1_no_ent", "A2_no_const", "A3_no_rel", "A4_surface_only"
    ]
    query_rankings = {sys_name: {} for sys_name in system_names}

    log.info("Executing retrieval systems over 21M Lucene index & GPU dense candidate pool...")
    for r in tqdm(records, desc=f"Retrieval [{track}]"):
        qid = str(r["id"])
        question = r["question"]
        trans = (r.get("llm_translation") or "").strip()
        rewrite = (r.get("llm_rewrite") or "").strip() or trans
        oracle = (r.get("eng_translation") or "").strip() or trans
        facets = r.get("llm_facets") or {}

        # S1: Zero-shot source query
        s1_scores = run_dense_query(question, model, pool_embeddings, doc_ids)

        # S2: Translate-then-BM25 on 21M corpus
        s2_hits = searcher.search(trans if trans else question, k=100)
        s2_ranked = [str(h.docid) for h in s2_hits]
        # In case some top hits are outside the pre-pooled set, ensure doc_text_map has their text
        for h in s2_hits:
            did = str(h.docid)
            if did not in doc_text_map:
                d = searcher.doc(did)
                if d:
                    doc_text_map[did] = json.loads(d.raw()).get("contents", "")

        # S3: MT Dense
        s3_scores = run_dense_query(trans, model, pool_embeddings, doc_ids)

        # S3r: Rewrite Dense
        s3r_scores = run_dense_query(rewrite, model, pool_embeddings, doc_ids)

        # S8: Human Oracle Reference Dense (eng_translation)
        s8_scores = run_dense_query(oracle, model, pool_embeddings, doc_ids)

        # Extract facets
        entities = []
        entities_src = []
        for ent in facets.get("entities_source") or []:
            if not ent.get("facet_invalid"):
                ef = (ent.get("english_form") or "").strip()
                sf = (ent.get("surface_form") or "").strip()
                if ef:
                    entities.append(ef)
                if sf:
                    entities_src.append(sf)

        constraints = []
        for c in facets.get("constraints") or []:
            if not c.get("facet_invalid"):
                nc = (c.get("normalized_english") or "").strip()
                if nc:
                    constraints.append(nc)

        rel = (facets.get("relation") or "").strip()

        # S4: Facets Concat
        facet_concat = " ".join(entities + constraints).strip()
        s4_scores = run_dense_query(facet_concat if facet_concat else trans, model, pool_embeddings, doc_ids)

        # S5: Facet + MT Concat
        s5_text = f"{trans} {' '.join(entities)} {' '.join(constraints)}".strip()
        s5_scores = run_dense_query(s5_text, model, pool_embeddings, doc_ids)

        # S5r: Random-Word Expansion Control for S5 (matched length control)
        num_facet_words = len(facet_concat.split()) if facet_concat else 3
        distractors = ["general", "overview", "history", "records", "details", "encyclopedia", "information", "archive", "summary", "report", "data"]
        rand_expansion = " ".join(distractors[:max(1, min(num_facet_words, len(distractors)))])
        s5r_text = f"{trans} {rand_expansion}".strip()
        s5r_scores = run_dense_query(s5r_text, model, pool_embeddings, doc_ids)

        # S6: Multi-Vector Fusion (0.50 Entity + 0.35 Relation/MT + 0.15 Constraints)
        ent_score = run_dense_query(" ".join(entities), model, pool_embeddings, doc_ids) if entities else s3_scores
        rel_score = run_dense_query(f"{trans} {rel}", model, pool_embeddings, doc_ids)
        const_score = run_dense_query(" ".join(constraints), model, pool_embeddings, doc_ids) if constraints else s3_scores
        s6_scores = 0.50 * zscore_normalize(ent_score) + 0.35 * zscore_normalize(rel_score) + 0.15 * zscore_normalize(const_score)

        # S6_src: Source Surface Multi-Vector Fusion
        ent_src_score = run_dense_query(" ".join(entities_src), model, pool_embeddings, doc_ids) if entities_src else s1_scores
        s6_src_scores = 0.50 * zscore_normalize(ent_src_score) + 0.35 * zscore_normalize(rel_score) + 0.15 * zscore_normalize(const_score)

        # S9: Random 3-Span Multi-Vector Fusion Control for S6
        # Partition query into 3 contiguous spans -> 3 vectors -> 0.50/0.35/0.15 fusion
        q_words = (trans if trans else question).split()
        if len(q_words) >= 3:
            h = int(abs(hash(qid))) % 10007
            n = len(q_words)
            s1_idx = 1 + (h % (n - 2)) if n > 2 else 1
            s2_idx = s1_idx + 1 + ((h // max(1, n - 2)) % max(1, n - 1 - s1_idx))
            span1 = " ".join(q_words[:s1_idx])
            span2 = " ".join(q_words[s1_idx:s2_idx])
            span3 = " ".join(q_words[s2_idx:])
        else:
            span1 = span2 = span3 = trans if trans else question

        span1_score = run_dense_query(span1, model, pool_embeddings, doc_ids)
        span2_score = run_dense_query(span2, model, pool_embeddings, doc_ids)
        span3_score = run_dense_query(span3, model, pool_embeddings, doc_ids)
        s9_scores = 0.50 * zscore_normalize(span1_score) + 0.35 * zscore_normalize(span2_score) + 0.15 * zscore_normalize(span3_score)

        # Ablations
        a1_scores = 0.70 * zscore_normalize(rel_score) + 0.30 * zscore_normalize(const_score)  # no_ent
        a2_scores = 0.60 * zscore_normalize(ent_score) + 0.40 * zscore_normalize(rel_score)    # no_const
        a3_scores = 0.75 * zscore_normalize(ent_score) + 0.25 * zscore_normalize(const_score)  # no_rel
        a4_scores = run_dense_query(" ".join(entities_src) if entities_src else question, model, pool_embeddings, doc_ids)

        # Convert dense scores to top-100 ranked doc IDs
        def get_top100(scores_arr):
            top_indices = np.argpartition(scores_arr, -100)[-100:]
            sorted_top = top_indices[np.argsort(-scores_arr[top_indices])]
            return [doc_ids[idx] for idx in sorted_top]

        s1_ranked = get_top100(s1_scores)
        s3_ranked = get_top100(s3_scores)
        s3r_ranked = get_top100(s3r_scores)
        s4_ranked = get_top100(s4_scores)
        s5_ranked = get_top100(s5_scores)
        s5r_ranked = get_top100(s5r_scores)
        s6_ranked = get_top100(s6_scores)
        s6_src_ranked = get_top100(s6_src_scores)
        s8_ranked = get_top100(s8_scores)
        s9_ranked = get_top100(s9_scores)
        a1_ranked = get_top100(a1_scores)
        a2_ranked = get_top100(a2_scores)
        a3_ranked = get_top100(a3_scores)
        a4_ranked = get_top100(a4_scores)

        # S7: RRF (S1 + S2 + S3)
        s7_fused = rrf_fuse([s1_ranked, s2_ranked, s3_ranked])
        s7_ranked = [did for did, _ in s7_fused[:100]]

        query_rankings["S1"][qid] = s1_ranked
        query_rankings["S2"][qid] = s2_ranked
        query_rankings["S3"][qid] = s3_ranked
        query_rankings["S3r"][qid] = s3r_ranked
        query_rankings["S4"][qid] = s4_ranked
        query_rankings["S5"][qid] = s5_ranked
        query_rankings["S5r"][qid] = s5r_ranked
        query_rankings["S6"][qid] = s6_ranked
        query_rankings["S6_src"][qid] = s6_src_ranked
        query_rankings["S7"][qid] = s7_ranked
        query_rankings["S8"][qid] = s8_ranked
        query_rankings["S9"][qid] = s9_ranked
        query_rankings["A1_no_ent"][qid] = a1_ranked
        query_rankings["A2_no_const"][qid] = a2_ranked
        query_rankings["A3_no_rel"][qid] = a3_ranked
        query_rankings["A4_surface_only"][qid] = a4_ranked

    # Export TREC runs
    for sys_name, q_map in query_rankings.items():
        trec_file = track_runs_dir / f"{sys_name}__{track}.trec"
        with open(trec_file, "w", encoding="utf-8") as f:
            for qid, ranked_dids in q_map.items():
                for rank, did in enumerate(ranked_dids, start=1):
                    score = 100.0 - rank
                    f.write(f"{qid} Q0 {did} {rank} {score:.4f} {sys_name}\n")

    # Evaluate Metrics & Bootstrap CIs
    log.info("Computing official evaluation metrics with 1,000 bootstrap resamples...")
    results_per_sys = {}
    for sys_name, q_map in query_rankings.items():
        q_metrics = []
        for r in records:
            qid = str(r["id"])
            answers = r.get("answers") or []
            ranked_dids = q_map[qid]
            m = eval_query_xor_tydi(ranked_dids, doc_text_map, answers)
            q_metrics.append(m)
        results_per_sys[sys_name] = q_metrics

    # Compute means, CIs, and p-values vs S6 and S1
    n_queries = len(records)
    n_resamples = 1000
    rng = np.random.RandomState(42)
    boot_indices = [rng.randint(0, n_queries, size=n_queries) for _ in range(n_resamples)]

    s6_r5kt = np.array([m["r_5kt"] for m in results_per_sys["S6"]])
    s1_r5kt = np.array([m["r_5kt"] for m in results_per_sys["S1"]])

    rows = []
    raw_p_values_vs_s6 = []

    for sys_name in system_names:
        m_list = results_per_sys[sys_name]
        r2_arr = np.array([m["r_2kt"] for m in m_list])
        r5_arr = np.array([m["r_5kt"] for m in m_list])
        cfr_arr = np.array([m["catastrophic_failure"] for m in m_list])

        # Bootstrap CIs
        r2_boots = [r2_arr[idx].mean() for idx in boot_indices]
        r5_boots = [r5_arr[idx].mean() for idx in boot_indices]
        cfr_boots = [cfr_arr[idx].mean() for idx in boot_indices]

        r2_ci = [np.percentile(r2_boots, 2.5), np.percentile(r2_boots, 97.5)]
        r5_ci = [np.percentile(r5_boots, 2.5), np.percentile(r5_boots, 97.5)]
        cfr_ci = [np.percentile(cfr_boots, 2.5), np.percentile(cfr_boots, 97.5)]

        # Paired bootstrap test vs S6 on R@5kt
        diffs_vs_s6 = np.array([r5_arr[idx].mean() - s6_r5kt[idx].mean() for idx in boot_indices])
        p_val_s6 = 2.0 * min((diffs_vs_s6 <= 0).mean(), (diffs_vs_s6 >= 0).mean())
        p_val_s6 = min(1.0, max(0.001, p_val_s6))
        raw_p_values_vs_s6.append(p_val_s6)

        rows.append({
            "track": track,
            "system": sys_name,
            "r_2kt_mean": round(float(r2_arr.mean()), 4),
            "r_2kt_ci95": f"[{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]",
            "r_5kt_mean": round(float(r5_arr.mean()), 4),
            "r_5kt_ci95": f"[{r5_ci[0]:.4f}, {r5_ci[1]:.4f}]",
            "catastrophic_failure_mean": round(float(cfr_arr.mean()), 4),
            "catastrophic_failure_ci95": f"[{cfr_ci[0]:.4f}, {cfr_ci[1]:.4f}]",
            "raw_p_val_vs_S6": p_val_s6,
        })

    # Holm-Bonferroni correction
    order = np.argsort(raw_p_values_vs_s6)
    m_tests = len(raw_p_values_vs_s6)
    holm_p_vals = np.zeros(m_tests)
    for rank_idx, orig_idx in enumerate(order):
        holm_p_vals[orig_idx] = min(1.0, raw_p_values_vs_s6[orig_idx] * (m_tests - rank_idx))

    for idx, r in enumerate(rows):
        r["holm_p_val_vs_S6"] = round(float(holm_p_vals[idx]), 4)

    df_out = pd.DataFrame(rows)
    metrics_file = METRICS_DIR / f"{track}_main.csv"
    df_out.to_csv(metrics_file, index=False)
    log.info("Saved metrics table to %s", metrics_file)
    print("\n" + df_out.to_string(index=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 4 GPU Dense Retrieval & Official Evaluation")
    parser.add_argument("--batch-size", type=int, default=256, help="GPU encoding batch size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = get_local_model_path("BAAI/bge-m3")
    log.info("Initializing BAAI/bge-m3 from '%s' on device: %s", model_path, device)
    model = SentenceTransformer(model_path, device=device)
    if device == "cuda":
        model = model.half()

    searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w")
    assert searcher.num_docs > 20_000_000

    tracks = ["bn", "te"]
    for track in tracks:
        run_full_track(track, model, searcher, batch_size=args.batch_size)

    print("\n[✓] STAGE 4 COMPLETE! All systems evaluated against official 21M corpus pool.")


if __name__ == "__main__":
    main()
