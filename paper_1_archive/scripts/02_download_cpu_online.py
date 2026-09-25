#!/usr/bin/env python3
"""
Stage 2: Download Data and Preprocess Query Sets (CPU, Online).

- Downloads XOR-TyDi (bn/te), mMARCO Hindi (hi), MS MARCO qrels & queries, MIRACL Hindi.
- Downloads HuggingFace embedding model snapshot for offline GPU stage.
- Prepares filtered/sampled dataset files in data/filtered/ with seed=42.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, assert_online, get_logger
from src.data_prep import filter_xor_tydi_bn_te, load_miracl_hi_monolingual, sample_mmarco_hi
from src.download import (
    download_hf_embedding_model,
    download_miracl_hi,
    download_mmarco_hi,
    download_msmarco_dev_queries,
    download_msmarco_qrels,
    download_xor_eval_script,
    download_xor_tydi,
    sanity_check_all_downloaded,
)
from src.utils_io import mark_done

log = get_logger("02_download")


def main():
    parser = argparse.ArgumentParser(description="Download datasets and models")
    parser.add_argument("--n", type=int, default=100, help="Number of queries to sample per track (-1 for full data)")
    parser.add_argument("--skip-hf-download", action="store_true", help="Skip downloading HuggingFace model")
    args = parser.parse_args()

    log.info("=== Stage 2: Download Datasets & HuggingFace Models (n=%d) ===", args.n)
    assert_online()

    # 1. Download raw datasets
    xor_path = download_xor_tydi()
    download_xor_eval_script()
    mmarco_hi_path = download_mmarco_hi()
    msmarco_qrels_path = download_msmarco_qrels()
    msmarco_en_queries_path = download_msmarco_dev_queries()
    miracl_topics, miracl_qrels = download_miracl_hi()

    # 2. Download HuggingFace embedding model snapshot
    if not args.skip_hf_download:
        download_hf_embedding_model()

    sanity_check_all_downloaded()

    # 3. Filter and sample queries per track
    log.info("Processing filtered query sets...")
    xor_filtered = filter_xor_tydi_bn_te(xor_path, n=args.n)
    mmarco_filtered = sample_mmarco_hi(
        queries_tsv=mmarco_hi_path,
        qrels_tsv=msmarco_qrels_path,
        en_queries_tsv=msmarco_en_queries_path,
        n=args.n,
        full_target_size=1000,
    )
    miracl_filtered = load_miracl_hi_monolingual(
        topics_tsv=miracl_topics,
        qrels_tsv=miracl_qrels,
        n=args.n,
    )

    # 4. Mark stage done
    for track in ["bn", "te", "hi", "hi_mono"]:
        mark_done(ROOT / "state", "stage02", track, {"n": args.n})

    log.info("Stage 2 completed successfully. Filtered query files ready in %s", DATA_DIR / "filtered")


if __name__ == "__main__":
    main()
