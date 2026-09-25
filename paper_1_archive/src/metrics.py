"""
Evaluation metrics, bootstrap confidence intervals, and significance testing.

Supports:
1. XOR-TyDi track: Answer containment (R@2kt, R@5kt, catastrophic-failure rate).
2. mMARCO Hindi track: Passage qrels (MRR@10, Recall@20, Recall@100, catastrophic-failure rate).
3. Monolingual controls: nDCG@10, Recall@100, MRR@10.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from src.utils_io import nfc


def _is_answer_contained(passage_text: str, answers: Sequence[str]) -> bool:
    norm_p = nfc(passage_text or "").lower()
    for ans in answers:
        norm_a = nfc(ans or "").lower().strip()
        if norm_a and norm_a in norm_p:
            return True
    return False


def eval_query_xor_tydi(
    retrieved_doc_ids: list[str],
    doc_text_map: dict[str, str],
    answers: list[str],
) -> dict[str, float]:
    """Computes R@2kt, R@5kt, and catastrophic failure for a single XOR-TyDi query."""
    if not answers:
        return {"r_2kt": 0.0, "r_5kt": 0.0, "catastrophic_failure": 1.0}

    r_2kt = 0.0
    r_5kt = 0.0
    total_chars = 0
    hit_any = False

    for doc_id in retrieved_doc_ids:
        text = doc_text_map.get(doc_id, "")
        total_chars += len(text)
        if _is_answer_contained(text, answers):
            hit_any = True
            if total_chars <= 2000 and r_2kt == 0.0:
                r_2kt = 1.0
            if total_chars <= 5000 and r_5kt == 0.0:
                r_5kt = 1.0

    return {
        "r_2kt": r_2kt,
        "r_5kt": r_5kt,
        "catastrophic_failure": 0.0 if hit_any else 1.0,
    }


def eval_query_mmarco(
    retrieved_doc_ids: list[str],
    relevant_passage_ids: list[str],
) -> dict[str, float]:
    """Computes MRR@10, Recall@20, Recall@100, and catastrophic failure for a single mMARCO query."""
    rel_set = set(str(pid) for pid in relevant_passage_ids)
    if not rel_set:
        return {"mrr_10": 0.0, "recall_20": 0.0, "recall_100": 0.0, "catastrophic_failure": 1.0}

    mrr_10 = 0.0
    hit_20 = 0.0
    hit_100 = 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if str(doc_id) in rel_set:
            if rank <= 10 and mrr_10 == 0.0:
                mrr_10 = 1.0 / rank
            if rank <= 20:
                hit_20 = 1.0
            if rank <= 100:
                hit_100 = 1.0

    return {
        "mrr_10": mrr_10,
        "recall_20": hit_20,
        "recall_100": hit_100,
        "catastrophic_failure": 0.0 if hit_100 > 0 else 1.0,
    }


def eval_query_ndcg_10(
    retrieved_doc_ids: list[str],
    relevant_passage_ids: list[str],
) -> dict[str, float]:
    """Computes nDCG@10 for monolingual controls."""
    rel_set = set(str(pid) for pid in relevant_passage_ids)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:10], start=1):
        if str(doc_id) in rel_set:
            dcg += 1.0 / math.log2(rank + 1)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel_set), 10) + 1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return {"ndcg_10": ndcg}


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Returns (mean, ci_lower, ci_upper) using paired/unpaired bootstrap."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100 * alpha))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return float(np.mean(values)), low, high


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> float:
    """Computes two-sided p-value using paired bootstrap hypothesis test."""
    if len(scores_a) != len(scores_b) or len(scores_a) == 0:
        return 1.0
    diff = scores_a - scores_b
    obs_diff = np.abs(np.mean(diff))
    rng = np.random.RandomState(seed)
    n = len(diff)
    cnt = 0
    for _ in range(n_boot):
        signs = rng.choice([-1.0, 1.0], size=n)
        sample_diff = np.abs(np.mean(diff * signs))
        if sample_diff >= obs_diff:
            cnt += 1
    return float(cnt) / n_boot
