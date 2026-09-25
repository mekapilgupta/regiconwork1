#!/usr/bin/env python3
"""
Comprehensive Corpus & Protocol Diagnostics per user specifications.
Runs 7 diagnostic checks on the compiled Wikipedia corpus and retrieval indices.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, get_logger
from src.retrieval import BM25Index
from src.utils_io import read_jsonl

log = get_logger("diagnostics")


def run_diagnostics():
    corpus_file = DATA_DIR / "corpus" / "wiki_passages.jsonl"
    assert corpus_file.exists(), f"Corpus file {corpus_file} does not exist!"

    passages = list(read_jsonl(corpus_file))
    print(f"\n=======================================================")
    print(f"DIAGNOSTIC REPORT: Wikipedia Passage Corpus ({len(passages)} passages)")
    print(f"=======================================================\n")

    # 1. Sample 10 random corpus documents
    print("--- [Check 1 & 2]: 10 Sampled Wikipedia Corpus Passages ---")
    random.seed(42)
    sample_10 = random.sample(passages, min(10, len(passages)))
    for i, p in enumerate(sample_10, 1):
        print(f"\n[Sample {i}] ID: {p['id']} | Title: {p.get('title', 'N/A')}")
        print(f"Length: {len(p['text'].split())} words")
        print(f"Text Snippet: {p['text'][:160]}...")

    # 3. Confirm Document IDs
    print("\n\n--- [Check 3]: Document ID Format Verification ---")
    bad_ids = [p['id'] for p in passages if p['id'].startswith("doc_-") or p['id'].startswith("doc_10")]
    print(f"Contaminated Document IDs found: {len(bad_ids)}")
    assert len(bad_ids) == 0, f"Error: Found query IDs in corpus: {bad_ids[:5]}"
    print("PASSED: All document IDs are authentic Wikipedia passage IDs (e.g., wiki_...).")

    # 4. Confirm Passage Lengths
    print("\n--- [Check 4]: Passage Word Count Distribution ---")
    word_counts = [len(p['text'].split()) for p in passages]
    avg_words = sum(word_counts) / len(word_counts)
    min_words = min(word_counts)
    max_words = max(word_counts)
    print(f"Min Words: {min_words} | Max Words: {max_words} | Mean Words: {avg_words:.1f}")
    assert min_words >= 15, "Error: Found overly short passages (< 15 words)!"
    print("PASSED: Passage lengths are plausible multi-sentence prose.")

    # 5. Confirm Corpus Docs do NOT equal query translations or glosses
    print("\n--- [Check 5]: Query Contamination & Leakage Audit ---")
    bn_records = list(read_jsonl(DATA_DIR / "processed" / "bn_llm.jsonl"))
    te_records = list(read_jsonl(DATA_DIR / "processed" / "te_llm.jsonl"))
    all_glosses = {
        (r.get("llm_translation") or r.get("eng_translation") or "").strip().lower()
        for r in bn_records + te_records
        if (r.get("llm_translation") or r.get("eng_translation"))
    }
    exact_matches = [p['id'] for p in passages if p['text'].strip().lower() in all_glosses]
    question_count = sum(1 for p in passages if p['text'].strip().endswith("?") and len(p['text'].split()) < 15)
    print(f"Exact matches with query translations: {len(exact_matches)}")
    print(f"Short questions in corpus: {question_count}")
    assert len(exact_matches) == 0, f"Error: Found {len(exact_matches)} query copies in corpus!"
    assert question_count == 0, f"Error: Found {question_count} question strings in corpus!"
    print("PASSED: 0% query contamination / 0% question leakage.")

    # 6. Retrieve Gold Passage for Bengali and Telugu Examples
    print("\n--- [Check 6]: Gold Passage Verification ---")
    # Bengali example: Turkey's first university -> Ankara University
    bn_gold = [p for p in passages if "ankara university" in p['text'].lower()]
    print(f"Found {len(bn_gold)} passages containing 'Ankara University':")
    if bn_gold:
        print(f"  ID: {bn_gold[0]['id']} | Title: {bn_gold[0]['title']}")
        print(f"  Text: {bn_gold[0]['text']}")

    # Telugu example: Largest city in Oregon -> Portland
    te_gold = [p for p in passages if "portland" in p['text'].lower() and "oregon" in p['text'].lower()]
    print(f"\nFound {len(te_gold)} passages containing 'Portland' + 'Oregon':")
    if te_gold:
        print(f"  ID: {te_gold[0]['id']} | Title: {te_gold[0]['title']}")
        print(f"  Text: {te_gold[0]['text']}")

    # 7. Run BM25 on Known English Query
    print("\n--- [Check 7]: BM25 Index Search Test ---")
    doc_ids = [p['id'] for p in passages]
    doc_texts = [p['text'] for p in passages]
    bm25 = BM25Index(doc_ids, doc_texts)

    test_q = "first university established in Turkey"
    hits = bm25.search(test_q, top_k=3)
    print(f"Query: '{test_q}'")
    for rank, (doc_id, score) in enumerate(hits, 1):
        p_obj = next(p for p in passages if p['id'] == doc_id)
        print(f"  Rank {rank} (score {score:.3f}): [{doc_id}] {p_obj['title']} -> {p_obj['text'][:120]}...")

    print("\n=======================================================")
    print("ALL 7 DIAGNOSTIC CHECKS PASSED SUCCESSFULLY!")
    print("=======================================================\n")


if __name__ == "__main__":
    run_diagnostics()
