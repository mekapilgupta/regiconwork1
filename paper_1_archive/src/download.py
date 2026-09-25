"""
All network downloads live here, and ONLY here. Run once on the CPU
(online) node via scripts/02_download_cpu_online.py. Nothing in this file
is imported by the GPU-stage scripts.

Everything is idempotent: if the target file/dir already exists (and,
where possible, a checksum/size sentinel matches), the download is skipped.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
import gzip
from pathlib import Path

from src.config import (
    DATA_DIR,
    HF_CACHE_DIR,
    PYSERINI_CACHE_DIR,
    STATE_DIR,
    get_logger,
    SETTINGS,
    get_pinned_embedding_model,
)

log = get_logger("download")

XOR_TYDI_URL = "https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_retrieve_eng_span_v1_1.jsonl"
XOR_EVAL_SCRIPT_URL = "https://raw.githubusercontent.com/AkariAsai/XORQA/main/evals/eval_xor_retrieve.py"
MMARCO_HI_DEV_SMALL_URL = "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/hindi_queries.dev.small.tsv"
MSMARCO_DEV_QUERIES_URL = "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/english_queries.dev.small.tsv"
MSMARCO_QRELS_DEV_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv"

# Passages and candidate pools
MSMARCO_PASSAGES_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
MIRACL_HI_DEV_URL = "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-hi/topics/topics.miracl-v1.0-hi-dev.tsv"
MIRACL_HI_QRELS_URL = "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-hi/qrels/qrels.miracl-v1.0-hi-dev.tsv"


def _download_file(url: str, dest: Path, timeout_s: int = 1200) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Already present, skipping: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s -> %s", url, dest)
    t0 = time.time()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.rename(dest)
    log.info("Done in %.1fs: %s (%d bytes)", time.time() - t0, dest, dest.stat().st_size)
    return dest


def download_xor_tydi() -> Path:
    return _download_file(XOR_TYDI_URL, DATA_DIR / "raw" / "xor_dev_retrieve_eng_span.jsonl")


def download_xor_eval_script() -> Path:
    dest = DATA_DIR.parent / "third_party" / "eval_xor_retrieve.py"
    return _download_file(XOR_EVAL_SCRIPT_URL, dest)


def download_mmarco_hi() -> Path:
    return _download_file(MMARCO_HI_DEV_SMALL_URL, DATA_DIR / "raw" / "mmarco_hi_dev_small_queries.tsv")


def download_msmarco_qrels() -> Path:
    return _download_file(MSMARCO_QRELS_DEV_URL, DATA_DIR / "raw" / "qrels.dev.small.tsv")


def download_msmarco_dev_queries() -> Path:
    return _download_file(MSMARCO_DEV_QUERIES_URL, DATA_DIR / "raw" / "queries.dev.small.tsv")


def download_miracl_hi() -> tuple[Path, Path]:
    q_path = _download_file(MIRACL_HI_DEV_URL, DATA_DIR / "raw" / "topics.miracl-v1.0-hi-dev.tsv")
    qrels_path = _download_file(MIRACL_HI_QRELS_URL, DATA_DIR / "raw" / "qrels.miracl-v1.0-hi-dev.tsv")
    return q_path, qrels_path


def download_hf_embedding_model() -> str:
    """Downloads whichever model was pinned by scripts/01_check_env.py
    (bge-m3 on GPU nodes, e5-small on CPU-only nodes) and its tokenizer,
    fully, so the GPU stage can load it with local_files_only=True."""
    import os
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    pin = get_pinned_embedding_model()
    model_id = pin["model_id"]
    from huggingface_hub import snapshot_download
    log.info("Downloading HF model snapshot: %s", model_id)
    path = snapshot_download(repo_id=model_id, cache_dir=str(HF_CACHE_DIR))
    log.info("HF model cached at %s", path)
    (STATE_DIR / "hf_model_downloaded.flag").write_text(model_id, encoding="utf-8")
    return path


def sanity_check_all_downloaded() -> dict:
    checks = {
        "xor_tydi_jsonl": (DATA_DIR / "raw" / "xor_dev_retrieve_eng_span.jsonl").exists(),
        "xor_eval_script": (DATA_DIR.parent / "third_party" / "eval_xor_retrieve.py").exists(),
        "mmarco_hi_queries": (DATA_DIR / "raw" / "mmarco_hi_dev_small_queries.tsv").exists(),
        "msmarco_qrels": (DATA_DIR / "raw" / "qrels.dev.small.tsv").exists(),
        "msmarco_en_queries": (DATA_DIR / "raw" / "queries.dev.small.tsv").exists(),
        "hf_model": (STATE_DIR / "hf_model_downloaded.flag").exists(),
    }
    log.info("Download sanity check: %s", checks)
    return checks
