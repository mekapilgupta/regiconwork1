"""
Retrieval engines: Dense vector search, BM25 sparse search, and Fusion methods (RRF & Convex).
"""
from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
import torch
from rank_bm25 import BM25Okapi

from src.config import get_logger
from src.utils_io import nfc

log = get_logger("retrieval")

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}\u0964\u0965]")


def tokenize(text: str) -> list[str]:
    text = nfc(text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    return text.split()


class BM25Index:
    def __init__(self, doc_ids: list[str], documents: list[str]):
        self.doc_ids = doc_ids
        self.corpus_size = len(documents)
        log.info("Building BM25 index over %d documents...", self.corpus_size)
        tokenized_corpus = [tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(self.doc_ids[i], float(scores[i])) for i in sorted_indices if scores[i] > 0]


class DenseIndex:
    def __init__(self, doc_ids: list[str], embeddings: np.ndarray, device: str = "cpu"):
        self.doc_ids = doc_ids
        self.num_docs = len(doc_ids)
        log.info("Building Dense index over %d documents (dim=%d)...", self.num_docs, embeddings.shape[1])
        self.device = device
        self.embeddings = torch.from_numpy(embeddings).to(device)

    def search(self, query_emb: np.ndarray, top_k: int = 100) -> list[tuple[str, float]]:
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
        q_tensor = torch.from_numpy(query_emb).to(self.device)
        # Cosine similarity assuming both query and doc embeddings are L2 normalized
        scores = torch.mm(q_tensor, self.embeddings.t()).squeeze(0)
        top_k = min(top_k, self.num_docs)
        top_scores, top_indices = torch.topk(scores, k=top_k)
        top_indices = top_indices.cpu().numpy()
        top_scores = top_scores.cpu().numpy()
        return [(self.doc_ids[i], float(s)) for i, s in zip(top_indices, top_scores)]


def reciprocal_rank_fusion(
    runs: Sequence[list[tuple[str, float]]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion: RRF_score(d) = sum_m (w_m / (k + rank_m(d)))
    where rank_m(d) is 1-indexed.
    """
    if not runs:
        return []
    if weights is None:
        weights = [1.0] * len(runs)

    doc_scores: dict[str, float] = defaultdict(float)
    for run, weight in zip(runs, weights):
        for rank, (doc_id, _) in enumerate(run, start=1):
            doc_scores[doc_id] += weight / (k + rank)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]


def convex_combination(
    run_a: list[tuple[str, float]],
    run_b: list[tuple[str, float]],
    alpha: float = 0.5,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    Normalizes scores min-max per run, then linearly combines:
    score(d) = alpha * score_a(d) + (1 - alpha) * score_b(d)
    """
    def _normalize(run: list[tuple[str, float]]) -> dict[str, float]:
        if not run:
            return {}
        scores = [s for _, s in run]
        min_s, max_s = min(scores), max(scores)
        denom = (max_s - min_s) if (max_s - min_s) > 1e-8 else 1.0
        return {doc_id: (s - min_s) / denom for doc_id, s in run}

    norm_a = _normalize(run_a)
    norm_b = _normalize(run_b)
    all_docs = set(norm_a.keys()).union(norm_b.keys())

    combined: dict[str, float] = {}
    for doc in all_docs:
        s_a = norm_a.get(doc, 0.0)
        s_b = norm_b.get(doc, 0.0)
        combined[doc] = alpha * s_a + (1.0 - alpha) * s_b

    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]
