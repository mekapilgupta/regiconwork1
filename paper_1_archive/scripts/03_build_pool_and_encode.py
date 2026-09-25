#!/usr/bin/env python3
"""
Stage 3: Shared Candidate Pool Construction & GPU Dense Encoding.

1. Candidate Pool Generation:
   - For each query in Bengali (n=312) and Telugu (n=238), runs BM25 top-100 searches against
     the official 21,015,324 Wikipedia passage index across multiple representations:
       * llm_translation
       * llm_rewrite
       * eng_translation (human oracle)
       * facets_concat
       * each english_form entity facet
       * each normalized_english constraint
   - Unions retrieved passages into a shared candidate pool per track.
   - Materializes authentic passage text strictly from LuceneSearcher.doc(docid).raw().
   - Enforces unique docid -> text hash invariant.
   - Computes and verifies pool_recall >= 0.60 against official answer spans.
   - Saves:
       * cache/pool/<track>/pool_ids.npy
       * cache/pool/<track>/pool_text.jsonl

2. GPU Dense Pool Encoding:
   - Encodes unique passages in the pool with BAAI/bge-m3 (fp16 on GPU).
   - Saves cache/pool/<track>/pool_emb.npy.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Pre-set Java & OpenAI dummy env
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("JAVA_HOME", "/home/kshitij22256/miniconda3/envs/slmtask/lib/jvm")
os.environ["PATH"] = f"/home/kshitij22256/miniconda3/envs/slmtask/bin:{os.environ.get('PATH', '')}"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

from src.config import DATA_DIR, get_logger
from src.metrics import _is_answer_contained
from src.utils_io import read_jsonl, write_jsonl

log = get_logger("03_build_pool")

POOL_CACHE_DIR = ROOT / "cache" / "pool"
POOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_candidate_pool(track: str, searcher: LuceneSearcher, top_k_per_variant: int = 100):
    log.info("=" * 60)
    log.info("Building Candidate Pool for Track: %s (BM25 top-%d on 21M corpus)", track.upper(), top_k_per_variant)
    log.info("=" * 60)

    track_pool_dir = POOL_CACHE_DIR / track
    track_pool_dir.mkdir(parents=True, exist_ok=True)

    proc_file = DATA_DIR / "processed" / f"{track}_llm.jsonl"
    assert proc_file.exists(), f"Processed query file not found: {proc_file}"
    records = list(read_jsonl(proc_file))
    log.info("Loaded %d queries for track '%s'", len(records), track)

    unique_docids = set()
    query_variants_map = {}

    for r in tqdm(records, desc=f"BM25 Search [{track}]"):
        qid = str(r["id"])
        variants = []

        # 1. LLM Translation
        trans = (r.get("llm_translation") or "").strip()
        if trans:
            variants.append(trans)

        # 2. LLM Rewrite
        rewrite = (r.get("llm_rewrite") or "").strip()
        if rewrite and rewrite != trans:
            variants.append(rewrite)

        # 3. Human Oracle Translation
        oracle = (r.get("eng_translation") or "").strip()
        if oracle and oracle not in (trans, rewrite):
            variants.append(oracle)

        # 4. Facets
        facets = r.get("llm_facets") or {}
        entities = []
        for ent in facets.get("entities_source") or []:
            if not ent.get("facet_invalid"):
                ef = (ent.get("english_form") or "").strip()
                if ef:
                    entities.append(ef)
                    variants.append(ef)

        constraints = []
        for c in facets.get("constraints") or []:
            if not c.get("facet_invalid"):
                nc = (c.get("normalized_english") or "").strip()
                if nc:
                    constraints.append(nc)
                    variants.append(nc)

        facet_concat = " ".join(entities + constraints).strip()
        if facet_concat and facet_concat not in variants:
            variants.append(facet_concat)

        query_variants_map[qid] = variants

        # Run BM25 for each variant
        for q_str in variants:
            hits = searcher.search(q_str, k=top_k_per_variant)
            for h in hits:
                unique_docids.add(str(h.docid))

    sorted_docids = sorted(list(unique_docids), key=lambda x: int(x) if x.isdigit() else x)
    log.info("Track '%s': Union candidate pool contains %d unique passages from 21M corpus.", track, len(sorted_docids))

    # Materialize authentic passage texts
    log.info("Materializing authentic passage texts from LuceneSearcher...")
    pool_passages = []
    docid_to_text = {}
    docid_to_hash = {}

    for docid in tqdm(sorted_docids, desc=f"Fetch Docs [{track}]"):
        doc = searcher.doc(str(docid))
        assert doc is not None, f"docid {docid} not found in LuceneSearcher!"
        raw_json = json.loads(doc.raw())
        content = raw_json.get("contents", "").strip()
        assert content, f"Empty contents for docid {docid}!"
        thash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if docid in docid_to_hash:
            assert docid_to_hash[docid] == thash, f"Hash mismatch for docid {docid}!"
        else:
            docid_to_hash[docid] = thash

        docid_to_text[docid] = content
        pool_passages.append({
            "id": docid,
            "text": content,
            "hash": thash[:16],
        })

    # Save pool passages and IDs
    pool_text_file = track_pool_dir / "pool_text.jsonl"
    write_jsonl(pool_text_file, pool_passages)

    pool_ids_file = track_pool_dir / "pool_ids.npy"
    np.save(pool_ids_file, np.array(sorted_docids))
    log.info("Saved %s and %s", pool_text_file, pool_ids_file)

    # Compute Pool Recall against gold answers
    log.info("Calculating official Pool Recall for track '%s'...", track)
    covered_queries = 0
    total_queries = len(records)

    for r in records:
        gold_answers = r.get("answers") or []
        query_covered = False
        # Check if any passage in the pool contains the answer
        for docid in sorted_docids:
            ptext = docid_to_text[docid]
            if _is_answer_contained(ptext, gold_answers):
                query_covered = True
                break
        if query_covered:
            covered_queries += 1

    pool_recall = covered_queries / total_queries if total_queries > 0 else 0.0
    log.info("============================================================")
    log.info("TRACK: %s | POOL RECALL CEILING: %.2f%% (%d / %d queries)",
             track.upper(), pool_recall * 100, covered_queries, total_queries)
    log.info("============================================================")

    assert pool_recall >= 0.60, (
        f"GATE 3 FAILED: pool_recall for {track} ({pool_recall:.4f}) is below required 0.60 threshold!"
    )
    return sorted_docids, pool_recall


def encode_pool_gpu(track: str, batch_size: int = 128):
    """Encodes the candidate pool with BAAI/bge-m3 on GPU."""
    import torch
    from sentence_transformers import SentenceTransformer

    track_pool_dir = POOL_CACHE_DIR / track
    pool_text_file = track_pool_dir / "pool_text.jsonl"
    out_emb_file = track_pool_dir / "pool_emb.npy"

    assert pool_text_file.exists(), f"Pool text file not found: {pool_text_file}"
    passages = list(read_jsonl(pool_text_file))
    texts = [p["text"] for p in passages]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Encoding %d passages for track '%s' using BAAI/bge-m3 on %s...", len(texts), track, device)

    model = SentenceTransformer("BAAI/bge-m3", device=device)
    if device == "cuda":
        model = model.half()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    np.save(out_emb_file, embeddings.astype(np.float32))
    log.info("Successfully saved %s (shape=%s)", out_emb_file, embeddings.shape)


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Pool Construction & GPU Dense Encoding")
    parser.add_argument("--step", choices=["pool", "encode", "all"], default="all",
                        help="Step to run: pool (BM25 union), encode (GPU dense), or all")
    parser.add_argument("--batch-size", type=int, default=128, help="GPU encoding batch size")
    args = parser.parse_args()

    searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w")
    assert searcher.num_docs > 20_000_000

    tracks = ["bn", "te"]

    if args.step in ("pool", "all"):
        for track in tracks:
            build_candidate_pool(track, searcher, top_k_per_variant=100)

    if args.step in ("encode", "all"):
        for track in tracks:
            encode_pool_gpu(track, batch_size=args.batch_size)

    print("\n[✓] STAGE 3 COMPLETE! Candidate pool constructed and encoded.")


if __name__ == "__main__":
    main()
