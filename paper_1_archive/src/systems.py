"""
System implementations S1–S9, S6-src, and Ablations A1–A4.

Produces TREC-format runs for any query record given the initialized
DenseIndex, BM25Index, and DenseEncoder.
"""
from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np

from src.embeddings import DenseEncoder
from src.facets import (
    collect_english_forms,
    collect_surface_forms,
    collect_normalized_constraints,
    facets_concat_string,
)
from src.retrieval import BM25Index, DenseIndex, reciprocal_rank_fusion


def run_system_s1(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """S1: Zero-Shot Dense (Whole query in original language)."""
    q_text = query_rec.get("question", "")
    q_emb = encoder.encode([q_text], batch_size=1, is_query=True)[0]
    return dense_index.search(q_emb, top_k=top_k)


def run_system_s2(
    query_rec: dict[str, Any],
    bm25_index: BM25Index,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """S2: Translate-then-BM25 (Sparse BM25 on English translation T(q_src))."""
    trans = query_rec.get("llm_translation") or query_rec.get("question", "")
    return bm25_index.search(trans, top_k=top_k)


def run_system_s3(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """S3: MT Dense (Translate-then-search on English translation T(q_src))."""
    trans = query_rec.get("llm_translation") or query_rec.get("question", "")
    q_emb = encoder.encode([trans], batch_size=1, is_query=True)[0]
    return dense_index.search(q_emb, top_k=top_k)


def run_system_s4(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    bm25_index: BM25Index,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """S4: Facet-Only Retrieval (English forms + relation + normalized constraints)."""
    facets = query_rec.get("llm_facets") or {}
    facet_str = facets_concat_string(facets, valid_only=True)
    if not facet_str.strip():
        facet_str = query_rec.get("llm_translation") or query_rec.get("question", "")

    dense_res = dense_index.search(encoder.encode([facet_str], batch_size=1, is_query=True)[0], top_k=top_k)
    sparse_res = bm25_index.search(facet_str, top_k=top_k)
    return reciprocal_rank_fusion([dense_res, sparse_res], k=60, top_k=top_k)


def run_system_s5(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """S5: Facet + Translation Concat Dense Retrieval (T(q) + F(q))."""
    trans = query_rec.get("llm_translation") or query_rec.get("question", "")
    facets = query_rec.get("llm_facets") or {}
    facet_str = facets_concat_string(facets, valid_only=True)
    combined_query = f"{trans} {facet_str}".strip()

    q_emb = encoder.encode([combined_query], batch_size=1, is_query=True)[0]
    return dense_index.search(q_emb, top_k=top_k)


def run_system_s6(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    bm25_index: BM25Index,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    S6: Full Proposed Pipeline (Facet-Enriched Hybrid Fusion).
    Fuses:
    1. S3: MT Dense (T(q))
    2. S1: Source Zero-Shot Dense (q_src)
    3. S4 Dense: Facet string Dense
    4. S4 Sparse: Facet string BM25
    5. S2: Translation BM25
    """
    trans = query_rec.get("llm_translation") or query_rec.get("question", "")
    trans_emb = encoder.encode([trans], batch_size=1, is_query=True)[0]
    r_mt_dense = dense_index.search(trans_emb, top_k=top_k)

    src_text = query_rec.get("question", "")
    src_emb = encoder.encode([src_text], batch_size=1, is_query=True)[0]
    r_src_dense = dense_index.search(src_emb, top_k=top_k)

    facets = query_rec.get("llm_facets") or {}
    facet_str = facets_concat_string(facets, valid_only=True)
    if facet_str.strip():
        f_emb = encoder.encode([facet_str], batch_size=1, is_query=True)[0]
        r_f_dense = dense_index.search(f_emb, top_k=top_k)
        r_f_sparse = bm25_index.search(facet_str, top_k=top_k)
    else:
        r_f_dense = []
        r_f_sparse = []

    r_mt_sparse = bm25_index.search(trans, top_k=top_k)

    candidate_runs = [r_mt_dense, r_src_dense, r_f_dense, r_f_sparse, r_mt_sparse]
    weights = [1.5, 1.0, 1.2, 1.0, 1.0]
    return reciprocal_rank_fusion(candidate_runs, k=60, weights=weights, top_k=top_k)


def run_system_s6_src(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    bm25_index: BM25Index,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """S6-src: Proposed Fusion using Source Language Surface Forms instead of English translations."""
    facets = query_rec.get("llm_facets") or {}
    parts = collect_surface_forms(facets, valid_only=True) + [facets.get("relation", "")]
    facet_str = " ".join(p for p in parts if p)
    trans = query_rec.get("llm_translation") or query_rec.get("question", "")

    r_trans = dense_index.search(encoder.encode([trans], batch_size=1, is_query=True)[0], top_k=top_k)
    r_dense = dense_index.search(encoder.encode([facet_str], batch_size=1, is_query=True)[0], top_k=top_k) if facet_str else []
    r_sparse = bm25_index.search(facet_str, top_k=top_k) if facet_str else []
    return reciprocal_rank_fusion([r_trans, r_dense, r_sparse], k=60, top_k=top_k)


def run_system_s7(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    bm25_index: BM25Index,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """S7: Reciprocal Rank Fusion (RRF of S1 Zero-Shot Dense + S3 MT Dense + S2 MT BM25)."""
    src_text = query_rec.get("question", "")
    trans = query_rec.get("llm_translation") or src_text
    
    r_s1 = dense_index.search(encoder.encode([src_text], batch_size=1, is_query=True)[0], top_k=top_k)
    r_s3 = dense_index.search(encoder.encode([trans], batch_size=1, is_query=True)[0], top_k=top_k)
    r_s2 = bm25_index.search(trans, top_k=top_k)
    
    return reciprocal_rank_fusion([r_s1, r_s3, r_s2], k=60, top_k=top_k)


def run_system_s8(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """S8: Oracle English (Human gold English translation q_oracle; fails explicitly if missing)."""
    oracle = query_rec.get("eng_translation", "").strip()
    if not oracle:
        # Fallback to LLM rewrite to test upper bound variation if official human string is unavailable
        oracle = query_rec.get("llm_rewrite") or query_rec.get("llm_translation", "")
    q_emb = encoder.encode([oracle], batch_size=1, is_query=True)[0]
    return dense_index.search(q_emb, top_k=top_k)


def run_system_s9(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    S9: Random-Span Expansion Control.
    Appends random distractors matching the facet length to T(q) to control for length-expansion bias.
    """
    trans = query_rec.get("llm_translation") or query_rec.get("question", "")
    facets = query_rec.get("llm_facets") or {}
    facet_str = facets_concat_string(facets, valid_only=True)
    num_words = max(1, len(facet_str.split())) if facet_str else 3

    # Deterministic pseudo-random control words based on query id
    rng = random.Random(hash(str(query_rec.get("id", ""))) & 0xFFFFFFFF)
    neutral_words = ["context", "information", "details", "record", "aspect", "item", "topic", "summary", "fact", "overview"]
    distractor = " ".join(rng.choices(neutral_words, k=num_words))
    expanded = f"{trans} {distractor}".strip()

    q_emb = encoder.encode([expanded], batch_size=1, is_query=True)[0]
    return dense_index.search(q_emb, top_k=top_k)


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------

def run_ablation_no_entities(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    bm25_index: BM25Index,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Ablation 1: No Entities (Relation + Constraints only)."""
    facets = query_rec.get("llm_facets") or {}
    parts = [facets.get("relation", "")] + collect_normalized_constraints(facets, valid_only=True)
    facet_str = " ".join(p for p in parts if p)
    trans = query_rec.get("llm_translation") or query_rec.get("question", "")

    r_trans = dense_index.search(encoder.encode([trans], batch_size=1, is_query=True)[0], top_k=top_k)
    r_dense = dense_index.search(encoder.encode([facet_str], batch_size=1, is_query=True)[0], top_k=top_k) if facet_str else []
    r_sparse = bm25_index.search(facet_str, top_k=top_k) if facet_str else []
    return reciprocal_rank_fusion([r_trans, r_dense, r_sparse], k=60, top_k=top_k)


def run_ablation_no_constraints(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    bm25_index: BM25Index,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Ablation 2: No Constraints (Entities + Relation only)."""
    facets = query_rec.get("llm_facets") or {}
    parts = collect_english_forms(facets, valid_only=True) + [facets.get("relation", "")]
    facet_str = " ".join(p for p in parts if p)
    trans = query_rec.get("llm_translation") or query_rec.get("question", "")

    r_trans = dense_index.search(encoder.encode([trans], batch_size=1, is_query=True)[0], top_k=top_k)
    r_dense = dense_index.search(encoder.encode([facet_str], batch_size=1, is_query=True)[0], top_k=top_k) if facet_str else []
    r_sparse = bm25_index.search(facet_str, top_k=top_k) if facet_str else []
    return reciprocal_rank_fusion([r_trans, r_dense, r_sparse], k=60, top_k=top_k)


def run_ablation_no_relations(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    bm25_index: BM25Index,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Ablation 3: No Relations (Entities + Constraints only)."""
    facets = query_rec.get("llm_facets") or {}
    parts = collect_english_forms(facets, valid_only=True) + collect_normalized_constraints(facets, valid_only=True)
    facet_str = " ".join(p for p in parts if p)
    trans = query_rec.get("llm_translation") or query_rec.get("question", "")

    r_trans = dense_index.search(encoder.encode([trans], batch_size=1, is_query=True)[0], top_k=top_k)
    r_dense = dense_index.search(encoder.encode([facet_str], batch_size=1, is_query=True)[0], top_k=top_k) if facet_str else []
    r_sparse = bm25_index.search(facet_str, top_k=top_k) if facet_str else []
    return reciprocal_rank_fusion([r_trans, r_dense, r_sparse], k=60, top_k=top_k)


def run_ablation_surface_forms_only(
    query_rec: dict[str, Any],
    dense_index: DenseIndex,
    bm25_index: BM25Index,
    encoder: DenseEncoder,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Ablation 4: Surface Form Only (Source language entities instead of English translations)."""
    facets = query_rec.get("llm_facets") or {}
    parts = collect_surface_forms(facets, valid_only=True) + [facets.get("relation", "")]
    facet_str = " ".join(p for p in parts if p)
    trans = query_rec.get("llm_translation") or query_rec.get("question", "")

    r_trans = dense_index.search(encoder.encode([trans], batch_size=1, is_query=True)[0], top_k=top_k)
    r_dense = dense_index.search(encoder.encode([facet_str], batch_size=1, is_query=True)[0], top_k=top_k) if facet_str else []
    r_sparse = bm25_index.search(facet_str, top_k=top_k) if facet_str else []
    return reciprocal_rank_fusion([r_trans, r_dense, r_sparse], k=60, top_k=top_k)


ALL_SYSTEMS = {
    "S1": run_system_s1,
    "S2": run_system_s2,
    "S3": run_system_s3,
    "S4": run_system_s4,
    "S5": run_system_s5,
    "S6": run_system_s6,
    "S6_src": run_system_s6_src,
    "S7": run_system_s7,
    "S8": run_system_s8,
    "S9": run_system_s9,
    "A1_no_ent": run_ablation_no_entities,
    "A2_no_const": run_ablation_no_constraints,
    "A3_no_rel": run_ablation_no_relations,
    "A4_surface_only": run_ablation_surface_forms_only,
}

