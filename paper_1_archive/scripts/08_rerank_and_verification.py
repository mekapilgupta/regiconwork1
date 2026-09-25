#!/usr/bin/env python3
"""
Step 2 & Step 3: Reranker Baselines and Constraint-Aware Verification Bonuses.

Step 2: Mandatory Rerank Baselines
- R-S1:   rerank S1 top-100 with bge-reranker-v2-m3 (translated query)
- R-S3:   rerank S3 top-100
- R-RRF:  rerank RRF(S1, S3, S2) top-100
- R-UNION: rerank plain union top-k {S1, S3, S2, S5} (no bonuses)

Step 3: Verification Bonuses (Evaluated individually and combined)
- R-UNION + A only (Answer-type compatibility)
- R-UNION + C only (Deterministic regex constraints: years/numbers/ordinals)
- R-UNION + G only (Lexical entity matching: exact english_form & source script)
- R-UNION + V only (Coverage of query content words)
- R-UNION + All (Tuned by 5-fold CV inside dev, evaluated out-of-fold)
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import DATA_DIR, METRICS_DIR, RUNS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("08_rerank_verify")


def load_doc_text_map(track: str) -> dict[str, str]:
    pool_file = ROOT / "cache" / "pool" / track / "pool_text.jsonl"
    passages = list(read_jsonl(pool_file))
    return {p["id"]: p["text"] for p in passages}


def load_trec_rankings(trec_file: Path) -> dict[str, list[str]]:
    rankings = defaultdict(list)
    with open(trec_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, did = parts[0], parts[2]
                rankings[qid].append(did)
    return rankings


# ---------------------------------------------------------
# Deterministic Verification Feature Extractors
# ---------------------------------------------------------
YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
NUM_PATTERN = re.compile(r"\b(\d+|first|second|third|fourth|fifth|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
MONTH_PATTERN = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)


def compute_features(query_obj: dict, passage_text: str) -> dict[str, float]:
    facets = query_obj.get("llm_facets") or {}
    trans = (query_obj.get("llm_translation") or "").strip()
    p_text_lower = passage_text.lower()

    # 1. Answer Type (A)
    ans_type = (facets.get("answer_type") or "").strip().upper()
    a_score = 0.0
    if ans_type == "DATE":
        if YEAR_PATTERN.search(passage_text) or MONTH_PATTERN.search(p_text_lower):
            a_score = 1.0
    elif ans_type in ("NUMBER", "QUANTITY"):
        if NUM_PATTERN.search(p_text_lower):
            a_score = 1.0
    elif ans_type == "PERSON":
        # Check for honorifics / person indicators
        if any(w in p_text_lower for w in ["born", "died", "mr.", "dr.", "king", "queen", "president", "author", "minister", "actor", "director"]):
            a_score = 1.0
    elif ans_type == "LOCATION":
        if any(w in p_text_lower for w in ["city", "country", "capital", "located", "province", "district", "state", "island", "river", "mountain"]):
            a_score = 1.0

    # 2. Deterministic Constraints (C)
    c_score = 0.0
    constraints = facets.get("constraints") or []
    if constraints:
        matches = 0
        valid_c = 0
        for c in constraints:
            if not c.get("facet_invalid"):
                nc = (c.get("normalized_english") or "").strip().lower()
                if nc:
                    valid_c += 1
                    # Regex for years/numbers or string match for words
                    if YEAR_PATTERN.search(nc):
                        year_match = YEAR_PATTERN.search(nc).group(0)
                        if year_match in passage_text:
                            matches += 1
                    elif nc in p_text_lower:
                        matches += 1
        if valid_c > 0:
            c_score = matches / valid_c

    # 3. Lexical Entities (G)
    g_score = 0.0
    entities = facets.get("entities_source") or []
    if entities:
        ent_matches = 0
        valid_e = 0
        for e in entities:
            if not e.get("facet_invalid"):
                ef = (e.get("english_form") or "").strip().lower()
                sf = (e.get("surface_form") or "").strip()
                if ef:
                    valid_e += 1
                    # Exact / canonical match of english form or source script match
                    if ef in p_text_lower or (sf and sf in passage_text):
                        ent_matches += 1
        if valid_e > 0:
            g_score = ent_matches / valid_e

    # 4. Coverage (V)
    # Content word overlap between translated query and passage
    stop_words = {"what", "which", "who", "whom", "where", "when", "why", "how", "is", "was", "are", "were", "the", "a", "an", "in", "of", "to", "for", "on", "by", "at", "did", "does"}
    q_tokens = [w for w in re.findall(r"\w+", trans.lower()) if w not in stop_words and len(w) > 2]
    if q_tokens:
        overlap = sum(1 for t in q_tokens if t in p_text_lower)
        v_score = overlap / len(q_tokens)
    else:
        v_score = 0.0

    return {"A": a_score, "C": c_score, "G": g_score, "V": v_score}


def rerank_pairs_batch(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    pairs: list[tuple[str, str]],
    batch_size: int = 64,
    device: str = "cuda"
) -> np.ndarray:
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            queries = [p[0] for p in batch]
            passages = [p[1] for p in batch]
            inputs = tokenizer(queries, passages, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Classification logit
            logits = outputs.logits.view(-1).float().cpu().numpy()
            all_scores.extend(logits)
    return np.array(all_scores, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Step 2 & Step 3: Reranker Baselines & Verification Bonuses")
    parser.add_argument("--batch-size", type=int, default=64, help="Reranker batch size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading bge-reranker-v2-m3 on %s...", device)

    # Find snapshot
    cache_hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-reranker-v2-m3" / "snapshots"
    model_dir = "BAAI/bge-reranker-v2-m3"
    if cache_hub.exists():
        for d in cache_hub.iterdir():
            if d.is_dir():
                model_dir = str(d)
                break

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    if device == "cuda":
        model.half()
    model.eval()

    tracks = ["bn", "te"]
    all_results = []

    for track in tracks:
        log.info("=" * 70)
        log.info("PROCESSING TRACK: %s", track.upper())
        log.info("=" * 70)

        proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        queries = list(read_jsonl(proc_file))
        doc_text_map = load_doc_text_map(track)

        # Load first-stage runs
        s1_rankings = load_trec_rankings(RUNS_DIR / track / f"S1__{track}.trec")
        s2_rankings = load_trec_rankings(RUNS_DIR / track / f"S2__{track}.trec")
        s3_rankings = load_trec_rankings(RUNS_DIR / track / f"S3__{track}.trec")
        s5_rankings = load_trec_rankings(RUNS_DIR / track / f"S5__{track}.trec")
        s7_rankings = load_trec_rankings(RUNS_DIR / track / f"S7__{track}.trec")

        # Prepare candidate lists for Step 2
        # R-S1: S1 top-100
        # R-S3: S3 top-100
        # R-RRF: S7 top-100 (which is RRF of S1+S2+S3)
        # R-UNION: Interleaved Union of {S1, S3, S2, S5} top-100 unique passages

        candidates_map = {"R-S1": {}, "R-S3": {}, "R-RRF": {}, "R-UNION": {}}
        features_map = defaultdict(lambda: defaultdict(dict))  # qid -> did -> {A, C, G, V}
        rerank_scores_map = defaultdict(dict)  # qid -> did -> rerank_score

        # Collect all unique (query, passage) pairs to rerank
        all_pairs_to_rerank = []
        pair_keys = []  # (qid, did)

        for q in queries:
            qid = str(q["id"])
            trans = (q.get("llm_translation") or "").strip()
            
            s1_top = s1_rankings.get(qid, [])[:100]
            s3_top = s3_rankings.get(qid, [])[:100]
            s7_top = s7_rankings.get(qid, [])[:100]

            # Interleaved union
            union_list = []
            seen = set()
            for r in range(100):
                for ranked in [s3_top, s1_top, s2_rankings.get(qid, []), s5_rankings.get(qid, [])]:
                    if r < len(ranked):
                        did = ranked[r]
                        if did not in seen:
                            seen.add(did)
                            union_list.append(did)
                        if len(union_list) >= 100:
                            break
                if len(union_list) >= 100:
                    break

            candidates_map["R-S1"][qid] = s1_top
            candidates_map["R-S3"][qid] = s3_top
            candidates_map["R-RRF"][qid] = s7_top
            candidates_map["R-UNION"][qid] = union_list

            # Pool all candidate passages for this query to rerank in one batch pass
            pooled_dids = list(dict.fromkeys(s1_top + s3_top + s7_top + union_list))
            for did in pooled_dids:
                p_text = doc_text_map.get(did, "")
                all_pairs_to_rerank.append((trans, p_text))
                pair_keys.append((qid, did))
                # Compute deterministic features
                features_map[qid][did] = compute_features(q, p_text)

        log.info("Reranking %d total (query, passage) pairs for track '%s'...", len(all_pairs_to_rerank), track)
        batch_logits = rerank_pairs_batch(tokenizer, model, all_pairs_to_rerank, batch_size=args.batch_size, device=device)

        for (qid, did), score in zip(pair_keys, batch_logits):
            rerank_scores_map[qid][did] = float(score)

        # ---------------------------------------------------------
        # Evaluate Step 2: Rerank Baselines
        # ---------------------------------------------------------
        def rank_by_scores(candidates_dict, bonus_weights=None):
            # bonus_weights: dict {'A': w_a, 'C': w_c, 'G': w_g, 'V': w_v}
            system_rankings = {}
            for q in queries:
                qid = str(q["id"])
                c_dids = candidates_dict[qid]
                scored = []
                for did in c_dids:
                    base_s = rerank_scores_map[qid].get(did, -999.0)
                    if bonus_weights:
                        feats = features_map[qid][did]
                        bonus = sum(bonus_weights[k] * feats[k] for k in bonus_weights)
                        final_s = base_s + bonus
                    else:
                        final_s = base_s
                    scored.append((did, final_s))
                scored.sort(key=lambda x: x[1], reverse=True)
                system_rankings[qid] = [did for did, _ in scored]
            return system_rankings

        def eval_rankings(system_rankings):
            metrics = []
            for q in queries:
                qid = str(q["id"])
                answers = q.get("answers", [])
                dids = system_rankings[qid]
                m = eval_query_xor_tydi(dids, doc_text_map, answers)
                metrics.append(m)
            r2 = np.mean([m["r_2kt"] for m in metrics])
            r5 = np.mean([m["r_5kt"] for m in metrics])
            cfr = np.mean([m["catastrophic_failure"] for m in metrics])
            return float(r2), float(r5), float(cfr), metrics

        # Evaluate Step 2 Baselines
        r_s1_rankings = rank_by_scores(candidates_map["R-S1"])
        r_s3_rankings = rank_by_scores(candidates_map["R-S3"])
        r_rrf_rankings = rank_by_scores(candidates_map["R-RRF"])
        r_union_rankings = rank_by_scores(candidates_map["R-UNION"])

        for name, r_map in [("R-S1", r_s1_rankings), ("R-S3", r_s3_rankings), ("R-RRF", r_rrf_rankings), ("R-UNION", r_union_rankings)]:
            r2, r5, cfr, _ = eval_rankings(r_map)
            all_results.append({
                "track": track, "system": name, "R@2kt": round(r2*100, 2), "R@5kt": round(r5*100, 2), "CFR": round(cfr*100, 2)
            })

        # ---------------------------------------------------------
        # Evaluate Step 3: Verification Bonuses on R-UNION
        # ---------------------------------------------------------
        # Single bonuses
        single_bonuses = [
            ("R-UNION + A (Answer-Type)", {"A": 1.0, "C": 0.0, "G": 0.0, "V": 0.0}),
            ("R-UNION + C (Constraints)", {"A": 0.0, "C": 1.0, "G": 0.0, "V": 0.0}),
            ("R-UNION + G (Lexical Entity)", {"A": 0.0, "C": 0.0, "G": 1.0, "V": 0.0}),
            ("R-UNION + V (Coverage)", {"A": 0.0, "C": 0.0, "G": 0.0, "V": 1.0}),
        ]
        for name, weights in single_bonuses:
            r_map = rank_by_scores(candidates_map["R-UNION"], bonus_weights=weights)
            r2, r5, cfr, _ = eval_rankings(r_map)
            all_results.append({
                "track": track, "system": name, "R@2kt": round(r2*100, 2), "R@5kt": round(r5*100, 2), "CFR": round(cfr*100, 2)
            })

        # 5-Fold Cross Validation for Combined Weights
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        q_indices = np.arange(len(queries))
        oof_rankings = {}

        candidate_grid = [0.0, 0.5, 1.0, 1.5]
        best_weights_per_fold = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(q_indices)):
            train_queries = [queries[i] for i in train_idx]
            val_queries = [queries[i] for i in val_idx]

            best_train_r5 = -1.0
            best_w = {"A": 0.5, "C": 1.0, "G": 1.0, "V": 0.5}

            # Coarse search on train fold
            for w_g in [0.5, 1.0, 1.5]:
                for w_c in [0.5, 1.0]:
                    for w_a in [0.0, 0.5]:
                        w_test = {"A": w_a, "C": w_c, "G": w_g, "V": 0.5}
                        # Evaluate on train queries
                        train_hits = []
                        for tq in train_queries:
                            qid = str(tq["id"])
                            ans = tq.get("answers", [])
                            c_dids = candidates_map["R-UNION"][qid]
                            scored = [(did, rerank_scores_map[qid].get(did, -999.0) + sum(w_test[k]*features_map[qid][did][k] for k in w_test)) for did in c_dids]
                            scored.sort(key=lambda x: x[1], reverse=True)
                            ranked_dids = [did for did, _ in scored]
                            m = eval_query_xor_tydi(ranked_dids, doc_text_map, ans)
                            train_hits.append(m["r_5kt"])
                        train_r5 = np.mean(train_hits)
                        if train_r5 > best_train_r5:
                            best_train_r5 = train_r5
                            best_w = w_test

            best_weights_per_fold.append(best_w)
            # Predict out-of-fold for val_queries
            for vq in val_queries:
                qid = str(vq["id"])
                c_dids = candidates_map["R-UNION"][qid]
                scored = [(did, rerank_scores_map[qid].get(did, -999.0) + sum(best_w[k]*features_map[qid][did][k] for k in best_w)) for did in c_dids]
                scored.sort(key=lambda x: x[1], reverse=True)
                oof_rankings[qid] = [did for did, _ in scored]

        r2_oof, r5_oof, cfr_oof, _ = eval_rankings(oof_rankings)
        all_results.append({
            "track": track, "system": "R-UNION + All (5-Fold CV OOF)", "R@2kt": round(r2_oof*100, 2), "R@5kt": round(r5_oof*100, 2), "CFR": round(cfr_oof*100, 2)
        })

    df_res = pd.DataFrame(all_results)
    out_file = METRICS_DIR / "step2_step3_rerank_verification.csv"
    df_res.to_csv(out_file, index=False)
    print("\n" + "=" * 80)
    print("STEP 2 & STEP 3: RERANK & VERIFICATION BONUSES RESULTS")
    print("=" * 80)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
