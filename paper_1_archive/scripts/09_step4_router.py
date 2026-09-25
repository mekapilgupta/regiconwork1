#!/usr/bin/env python3
"""
Step 4: Query-Conditioned Routing with 5-Fold Cross Validation x 10 Repeats.

Features (<= 8 interpretable features):
1. query_length: Character length of the source query.
2. word_count: Number of whitespace tokens in the source query.
3. has_year: Deterministic indicator if query contains a 4-digit year.
4. num_entities: Count of detected entities in source question.
5. num_constraints: Count of detected constraints.
6. bm25_max_score: Max BM25 score of translated query over English Lucene index.
7. mt_length_ratio: Ratio of translated query character length to source query character length.
8. is_telugu: Binary indicator (1 for Telugu, 0 for Bengali).

Models evaluated:
1. Best Single per language (S3 for bn, S1 for te)
2. Language-Only Router (bn -> S3, te -> S1)
3. Uniform RRF (S1 + S2 + S3)
4. Learned Router (Logistic Regression, 5-Fold CV repeated 10x, pooled out-of-fold predictions)
5. Oracle Router ({S1, S3} per-query optimal)

Decision: If Learned Router <= Language-Only + 1 pt, report Language-Only as the method.
"""
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
from pyserini.search.lucene import LuceneSearcher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from src.config import DATA_DIR, METRICS_DIR, RUNS_DIR, get_logger
from src.metrics import _is_answer_contained, eval_query_xor_tydi
from src.utils_io import read_jsonl

log = get_logger("09_step4_router")


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


YEAR_REGEX = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")


def extract_router_features(queries: list[dict], searcher: LuceneSearcher) -> np.ndarray:
    feature_matrix = []
    for q in queries:
        q_src = q.get("question", "")
        q_trans = q.get("llm_translation", "")
        lang = q.get("lang", "")
        facets = q.get("llm_facets") or {}

        # 1. query_length
        f1 = len(q_src)
        # 2. word_count
        f2 = len(q_src.split())
        # 3. has_year
        f3 = 1.0 if YEAR_REGEX.search(q_src) or YEAR_REGEX.search(q_trans) else 0.0
        # 4. num_entities
        entities = facets.get("entities_source") or []
        f4 = float(len([e for e in entities if not e.get("facet_invalid")]))
        # 5. num_constraints
        constraints = facets.get("constraints") or []
        f5 = float(len([c for c in constraints if not c.get("facet_invalid")]))
        # 6. bm25_max_score
        hits = searcher.search(q_trans if q_trans else q_src, k=5)
        f6 = float(hits[0].score) if hits else 0.0
        # 7. mt_length_ratio
        f7 = float(len(q_trans) / max(1, len(q_src)))
        # 8. is_telugu
        f8 = 1.0 if lang == "te" else 0.0

        feature_matrix.append([f1, f2, f3, f4, f5, f6, f7, f8])
    return np.array(feature_matrix, dtype=np.float32)


def main():
    print("=" * 80)
    print("STEP 4: QUERY-CONDITIONED ROUTING (5-Fold CV x 10 Repeats)")
    print("=" * 80)

    searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w")

    # Load all queries from both tracks
    bn_queries = list(read_jsonl(DATA_DIR / "processed" / "bn_llm.jsonl"))
    te_queries = list(read_jsonl(DATA_DIR / "processed" / "te_llm.jsonl"))
    all_queries = bn_queries + te_queries

    bn_doc_map = load_doc_text_map("bn")
    te_doc_map = load_doc_text_map("te")
    all_doc_map = {**bn_doc_map, **te_doc_map}

    # Load S1, S3, S7 (RRF) runs
    s1_bn = load_trec_rankings(RUNS_DIR / "bn" / "S1__bn.trec")
    s3_bn = load_trec_rankings(RUNS_DIR / "bn" / "S3__bn.trec")
    s7_bn = load_trec_rankings(RUNS_DIR / "bn" / "S7__bn.trec")

    s1_te = load_trec_rankings(RUNS_DIR / "te" / "S1__te.trec")
    s3_te = load_trec_rankings(RUNS_DIR / "te" / "S3__te.trec")
    s7_te = load_trec_rankings(RUNS_DIR / "te" / "S7__te.trec")

    s1_all = {**s1_bn, **s1_te}
    s3_all = {**s3_bn, **s3_te}
    s7_all = {**s7_bn, **s7_te}

    # Compute binary performance per query: y = 1 if S3 (MT) is better than S1, 0 otherwise
    s1_hits_5k = []
    s3_hits_5k = []
    rrf_hits_5k = []
    lang_router_hits_5k = []
    oracle_router_hits_5k = []
    labels_s3_better = []

    for q in all_queries:
        qid = str(q["id"])
        ans = q.get("answers", [])
        lang = q.get("lang", "")

        m1 = eval_query_xor_tydi(s1_all.get(qid, []), all_doc_map, ans)["r_5kt"]
        m3 = eval_query_xor_tydi(s3_all.get(qid, []), all_doc_map, ans)["r_5kt"]
        m7 = eval_query_xor_tydi(s7_all.get(qid, []), all_doc_map, ans)["r_5kt"]

        s1_hits_5k.append(m1)
        s3_hits_5k.append(m3)
        rrf_hits_5k.append(m7)

        # Language-only router: bn -> S3, te -> S1
        lang_choice = m3 if lang == "bn" else m1
        lang_router_hits_5k.append(lang_choice)

        # Oracle router: max(S1, S3)
        oracle_router_hits_5k.append(max(m1, m3))

        # Target label: 1 if S3 > S1 (S3 hit and S1 missed), 0 if S1 > S3, 0.5/default to language choice if tied
        if m3 > m1:
            labels_s3_better.append(1)
        elif m1 > m3:
            labels_s3_better.append(0)
        else:
            labels_s3_better.append(1 if lang == "bn" else 0)

    X = extract_router_features(all_queries, searcher)
    y = np.array(labels_s3_better, dtype=np.int32)
    s1_arr = np.array(s1_hits_5k)
    s3_arr = np.array(s3_hits_5k)

    log.info("Extracted %d feature vectors of dim %d.", X.shape[0], X.shape[1])

    # 5-fold CV repeated 10x
    rskf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    learned_router_repeats = []

    for repeat_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(C=0.5, penalty="l2", random_state=42, max_iter=500)
        clf.fit(X_train_scaled, y_train)

        preds = clf.predict(X_test_scaled)
        # If pred == 1 => route to S3, if pred == 0 => route to S1
        routed_hits = np.where(preds == 1, s3_arr[test_idx], s1_arr[test_idx])
        learned_router_repeats.append(np.mean(routed_hits))

    learned_router_r5 = float(np.mean(learned_router_repeats)) * 100
    learned_router_std = float(np.std(learned_router_repeats)) * 100

    s1_overall = float(np.mean(s1_hits_5k)) * 100
    s3_overall = float(np.mean(s3_hits_5k)) * 100
    rrf_overall = float(np.mean(rrf_hits_5k)) * 100
    lang_overall = float(np.mean(lang_router_hits_5k)) * 100
    oracle_overall = float(np.mean(oracle_router_hits_5k)) * 100

    results = [
        {"Model / Router": "S1 (Zero-Shot Dense)", "R@5kt (%)": f"{s1_overall:.2f}%", "Delta vs Lang-Only (pts)": f"{s1_overall - lang_overall:+.2f}"},
        {"Model / Router": "S3 (MT Dense)", "R@5kt (%)": f"{s3_overall:.2f}%", "Delta vs Lang-Only (pts)": f"{s3_overall - lang_overall:+.2f}"},
        {"Model / Router": "Uniform RRF (S1+S2+S3)", "R@5kt (%)": f"{rrf_overall:.2f}%", "Delta vs Lang-Only (pts)": f"{rrf_overall - lang_overall:+.2f}"},
        {"Model / Router": "Language-Only Router (bn->S3, te->S1)", "R@5kt (%)": f"{lang_overall:.2f}%", "Delta vs Lang-Only (pts)": f"+0.00 (Baseline)"},
        {"Model / Router": "Learned Router (Logistic Reg, 10x 5-Fold OOF)", "R@5kt (%)": f"{learned_router_r5:.2f}% ± {learned_router_std:.2f}%", "Delta vs Lang-Only (pts)": f"{learned_router_r5 - lang_overall:+.2f}"},
        {"Model / Router": "Oracle Router ({S1, S3} Per-Query)", "R@5kt (%)": f"{oracle_overall:.2f}%", "Delta vs Lang-Only (pts)": f"{oracle_overall - lang_overall:+.2f}"},
    ]

    df_router = pd.DataFrame(results)
    out_csv = METRICS_DIR / "step4_router_results.csv"
    df_router.to_csv(out_csv, index=False)
    print("\n" + df_router.to_string(index=False) + "\n")

    print("=== DECISION & METHOD RECOMMENDATION ===")
    gain_over_lang = learned_router_r5 - lang_overall
    if gain_over_lang <= 1.0:
        print(f"Learned router gain over Language-Only is {gain_over_lang:+.2f} pts (<= +1.0 pt).")
        print("-> Decision: Report LANGUAGE-ONLY ROUTER as the primary method in the paper (simple, transparent, $0).")
    else:
        print(f"Learned router gain over Language-Only is {gain_over_lang:+.2f} pts (> +1.0 pt).")
        print("-> Decision: Report LEARNED ROUTER alongside Language-Only baseline.")


if __name__ == "__main__":
    main()
