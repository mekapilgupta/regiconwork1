"""
Leakage audit (spec §3). Deterministic, runs before retrieval.

leak = any gold answer appears in translation/rewrite/any facet string
       AND not in the original question (after normalization).
fuzzy_leak = >=0.8 token-overlap between a gold answer and any
             translation/rewrite/facet string, when the exact check misses.

Also implements the mMARCO-track variant (spec 1b): no answer strings exist
there (graded relevance qrels), so leakage is redefined as "a facet contains
a 4+-token span copied from the gold passage that is absent from the query".
"""
from __future__ import annotations

import re
import string
from typing import Any, Iterable

from src.utils_io import nfc

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}\u0964\u0965]")  # incl. Devanagari danda


def normalize_text(s: str) -> str:
    s = nfc(s or "")
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> list[str]:
    return normalize_text(s).split()


def _token_overlap_ratio(a: str, b_tokens_set: set[str]) -> float:
    a_tokens = _tokens(a)
    if not a_tokens:
        return 0.0
    hit = sum(1 for t in a_tokens if t in b_tokens_set)
    return hit / len(a_tokens)


def _collect_facet_strings(facets: dict[str, Any]) -> list[str]:
    strs = []
    for ent in facets.get("entities_source", []) or []:
        strs += [ent.get("surface_form", ""), ent.get("romanized_form", ""), ent.get("english_form", "")]
    strs.append(facets.get("relation", "") or "")
    for c in facets.get("constraints", []) or []:
        strs += [c.get("text_source", ""), c.get("normalized_english", "")]
    return [s for s in strs if s]


def audit_query_answer_containment(
    query_id: str,
    lang: str,
    question: str,
    translation: str,
    rewrite: str,
    facets: dict[str, Any],
    answers: Iterable[str],
) -> dict[str, Any]:
    """XOR-TyDi / Mr.TyDi-style audit (answer-string based)."""
    q_norm = normalize_text(question)
    candidate_strings = {
        "translation": translation or "",
        "rewrite": rewrite or "",
    }
    for i, fs in enumerate(_collect_facet_strings(facets)):
        candidate_strings[f"facet_{i}"] = fs

    leak = False
    fuzzy_leak = False
    leaked_fields: list[str] = []
    fuzzy_fields: list[str] = []

    for field_name, text in candidate_strings.items():
        text_norm = normalize_text(text)
        text_tokens_set = set(_tokens(text))
        for ans in answers:
            ans_norm = normalize_text(ans)
            if not ans_norm:
                continue
            in_original = ans_norm in q_norm
            in_field = ans_norm in text_norm
            if in_field and not in_original:
                leak = True
                leaked_fields.append(field_name)
            elif not in_original:
                ratio = _token_overlap_ratio(ans, text_tokens_set)
                if ratio >= 0.8:
                    fuzzy_leak = True
                    fuzzy_fields.append(field_name)

    return {
        "id": query_id,
        "lang": lang,
        "leak": leak,
        "fuzzy_leak": fuzzy_leak,
        "leaked_fields": sorted(set(leaked_fields)),
        "fuzzy_fields": sorted(set(fuzzy_fields)),
    }


def audit_query_passage_span(
    query_id: str,
    lang: str,
    question: str,
    facets: dict[str, Any],
    gold_passage_text: str,
    min_span_tokens: int = 4,
) -> dict[str, Any]:
    """mMARCO-track audit (spec §1b): flag if a facet string contains a
    4+-token contiguous span from the gold passage that is absent from the
    query itself."""
    q_tokens = _tokens(question)
    passage_tokens = _tokens(gold_passage_text)
    passage_ngrams = set()
    for i in range(len(passage_tokens) - min_span_tokens + 1):
        passage_ngrams.add(tuple(passage_tokens[i:i + min_span_tokens]))

    leak = False
    leaked_fields = []
    for field_name, text in {**{f"facet_{i}": s for i, s in enumerate(_collect_facet_strings(facets))}}.items():
        f_tokens = _tokens(text)
        for i in range(len(f_tokens) - min_span_tokens + 1):
            span = tuple(f_tokens[i:i + min_span_tokens])
            if span in passage_ngrams and span not in {
                tuple(q_tokens[j:j + min_span_tokens]) for j in range(len(q_tokens) - min_span_tokens + 1)
            }:
                leak = True
                leaked_fields.append(field_name)

    return {
        "id": query_id, "lang": lang, "leak": leak,
        "leaked_fields": sorted(set(leaked_fields)),
    }
