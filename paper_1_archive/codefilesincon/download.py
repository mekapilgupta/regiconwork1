"""
All network downloads live here, and ONLY here. Run once on the CPU
(online) node via scripts/02_download_cpu_online.py. Nothing in this file
is imported by the GPU-stage scripts.

Everything is idempotent: if the target file/dir already exists (and,
where possible, a checksum/size sentinel matches), the download is skipped.
"""
from __future__ import annotations

import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

from src.config import DATA_DIR, HF_CACHE_DIR, PYSERINI_CACHE_DIR, STATE_DIR, get_logger, SETTINGS, get_pinned_embedding_model

log = get_logger("download")

XOR_TYDI_URL = "https://raw.githubusercontent.com/AkariAsai/XORQA/main/data/xor_dev_retrieve_eng_span.jsonl"
XOR_EVAL_SCRIPT_URL = "https://raw.githubusercontent.com/AkariAsai/XORQA/main/evals/eval_xor_retrieve.py"
MMARCO_HI_DEV_SMALL_URL = "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/hi_queries.dev.small.tsv"
MSMARCO_QRELS_DEV_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv"


def _download_file(url: str, dest: Path, timeout_s: int = 900) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Already present, skipping: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s -> %s", url, dest)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "clir-pilot/1.0"})
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


def download_pyserini_prebuilt_index(index_name: str, deadline_min: int = 15) -> tuple[bool, str]:
    """
    Uses pyserini's own prebuilt-index fetch mechanism (it manages its own
    cache dir under PYSERINI_CACHE_DIR / .cache/pyserini). Returns
    (success, message). On failure or timeout the caller (script 02) is
    responsible for the Mr.TyDi monolingual fallback per spec §1.
    """
    import os
    os.environ.setdefault("PYSERINI_CACHE", str(PYSERINI_CACHE_DIR))

    log.info("Fetching Pyserini prebuilt index '%s' (deadline %d min)...", index_name, deadline_min)
    t0 = time.time()
    try:
        from pyserini.search.lucene import LuceneSearcher
        proc_result = {}

        def _target():
            try:
                proc_result["searcher"] = LuceneSearcher.from_prebuilt_index(index_name)
            except Exception as e:  # noqa: BLE001
                proc_result["error"] = str(e)

        import threading
        th = threading.Thread(target=_target, daemon=True)
        th.start()
        th.join(timeout=deadline_min * 60)
        elapsed = time.time() - t0
        if th.is_alive():
            return False, f"timed out after {elapsed:.0f}s (> {deadline_min} min)"
        if "error" in proc_result:
            return False, proc_result["error"]
        log.info("Index '%s' ready in %.0fs", index_name, elapsed)
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


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
        "hf_model": (STATE_DIR / "hf_model_downloaded.flag").exists(),
    }
    log.info("Download sanity check: %s", checks)
    return checks
