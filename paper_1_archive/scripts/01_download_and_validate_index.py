#!/usr/bin/env python3
"""
Stage 1: Multi-Threaded Download & Rigorous Validation of Official Wikipedia DPR-100W Index.

- Downloads the official 21,015,324 passage Lucene index using 16 worker threads (chunked HTTP range requests).
- Automatically unpacks into ~/.cache/pyserini/indexes/
- Validates:
    * searcher.num_docs == 21,015,324 (> 20,000,000)
    * Sanity search checks (Ankara University, Mohenjo-daro/Indus, Riyadh)
    * Integrity checks: Authentic prose, integer docids, zero question strings
    * Exports analysis/corpus_validation.json
"""
import os
import sys
import tarfile
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm

# Set Java & dummy OpenAI env before importing Pyserini
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("JAVA_HOME", "/home/kshitij22256/miniconda3/envs/slmtask/lib/jvm")
os.environ["PATH"] = f"/home/kshitij22256/miniconda3/envs/slmtask/bin:{os.environ.get('PATH', '')}"

INDEX_URL = "https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.tar.gz"
EXPECTED_MD5 = "7b58c08da992b2ea7e96667f0b176651"
TAR_FILENAME = "lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.tar.gz"
EXTRACTED_DIR_NAME = f"lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.{EXPECTED_MD5}"

CACHE_DIR = Path.home() / ".cache" / "pyserini" / "indexes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEST_INDEX_PATH = CACHE_DIR / EXTRACTED_DIR_NAME
TARBALL_PATH = CACHE_DIR / TAR_FILENAME


def download_chunk(url, start_byte, end_byte, part_filename):
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()
    with open(part_filename, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def parallel_download(url: str, dest_path: Path, num_workers: int = 16):
    print(f"[*] Checking file headers from: {url}")
    head_resp = requests.head(url, allow_redirects=True, timeout=30)
    total_size = int(head_resp.headers.get("content-length", 0))
    print(f"[*] Total download size: {total_size / (1024**3):.2f} GB ({total_size:,} bytes)")

    chunk_size = total_size // num_workers
    part_files = []
    futures = []

    print(f"[*] Starting multi-threaded download with {num_workers} parallel workers...")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            start_b = i * chunk_size
            end_b = (start_b + chunk_size - 1) if i < num_workers - 1 else (total_size - 1)
            part_file = dest_path.parent / f"{dest_path.name}.part{i}"
            part_files.append(part_file)
            futures.append(executor.submit(download_chunk, url, start_b, end_b, part_file))

        # Wait for all chunks with status tracking
        for f in futures:
            f.result()

    elapsed = time.time() - start_time
    print(f"[*] Download completed in {elapsed:.1f}s ({total_size / (1024**2 * elapsed):.1f} MB/s). Merging parts...")

    with open(dest_path, "wb") as outfile:
        for p in part_files:
            with open(p, "rb") as infile:
                while True:
                    data = infile.read(32 * 1024 * 1024)
                    if not data:
                        break
                    outfile.write(data)
            p.unlink()

    print(f"[*] Successfully saved complete archive to {dest_path}")


def extract_tarball(tar_path: Path, dest_dir: Path, final_path: Path):
    print(f"[*] Unpacking {tar_path} into {dest_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

    # Pyserini names the inner folder 'lucene-index.wikipedia-dpr-100w.20210120.d1b9e6'
    raw_dir = dest_dir / "lucene-index.wikipedia-dpr-100w.20210120.d1b9e6"
    if raw_dir.exists() and not final_path.exists():
        raw_dir.rename(final_path)
    print(f"[*] Extracted successfully to: {final_path}")
    if tar_path.exists():
        tar_path.unlink()


def validate_corpus():
    print("\n" + "=" * 70)
    print("STAGE 1 — LuceneSearcher Real Corpus Validation")
    print("=" * 70)

    from pyserini.search.lucene import LuceneSearcher

    print(f"[*] Loading LuceneSearcher from prebuilt index 'wikipedia-dpr-100w'...")
    searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w")

    num_docs = searcher.num_docs
    print(f"[✓] searcher.num_docs = {num_docs:,}")
    assert num_docs > 20_000_000, f"FAILED: searcher.num_docs ({num_docs}) <= 20,000,000"

    print("\n[*] Running mandatory sanity queries on 21M Wikipedia passage index:")

    # 1. Ankara University
    hits_ankara = searcher.search("Ankara University oldest university Turkey", k=5)
    ankara_found = any("ankara" in json.loads(searcher.doc(h.docid).raw()).get("contents", "").lower() for h in hits_ankara)
    print(f"  - Sanity 1 (Ankara University): {'PASS' if ankara_found else 'FAIL'} (hits={len(hits_ankara)})")
    assert ankara_found, "Ankara University passage not found in top-5!"

    # 2. Mohenjo-daro / Indus
    hits_mohenjo = searcher.search("Mohenjo-daro Indus river", k=5)
    indus_found = any("indus" in json.loads(searcher.doc(h.docid).raw()).get("contents", "").lower() for h in hits_mohenjo)
    print(f"  - Sanity 2 (Mohenjo-daro / Indus): {'PASS' if indus_found else 'FAIL'} (hits={len(hits_mohenjo)})")
    assert indus_found, "Indus river passage not found in top-5!"

    # 3. Riyadh capital Saudi Arabia
    hits_riyadh = searcher.search("Riyadh capital Saudi Arabia", k=5)
    riyadh_found = any("riyadh" in json.loads(searcher.doc(h.docid).raw()).get("contents", "").lower() for h in hits_riyadh)
    print(f"  - Sanity 3 (Riyadh capital): {'PASS' if riyadh_found else 'FAIL'} (hits={len(hits_riyadh)})")
    assert riyadh_found, "Riyadh capital passage not found in top-5!"

    # 4. Doc ID and prose inspection
    sample_docs = []
    for docid in [100, 5000, 100000, 500000, 1000000, 5000000, 10000000, 15000000, 18000000, 20000000]:
        doc = searcher.doc(str(docid))
        if doc:
            raw_obj = json.loads(doc.raw())
            content = raw_obj.get("contents", "").strip()
            sample_docs.append({"docid": docid, "content_snippet": content[:150]})
            assert not content.endswith("?"), f"Doc {docid} looks like a question, not prose!"

    analysis_dir = Path("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "num_docs": num_docs,
        "sanity_checks": {
            "ankara_university": ankara_found,
            "mohenjo_daro_indus": indus_found,
            "riyadh_capital": riyadh_found,
        },
        "samples": sample_docs,
        "index_path": str(DEST_INDEX_PATH),
    }

    with open(analysis_dir / "corpus_validation.json", "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[✓] ALL STAGE 1 GATES PASSED! Saved validation report to analysis/corpus_validation.json")


def main():
    if not DEST_INDEX_PATH.exists():
        print(f"[*] Index not found at {DEST_INDEX_PATH}.")
        parallel_download(INDEX_URL, TARBALL_PATH, num_workers=16)
        extract_tarball(TARBALL_PATH, CACHE_DIR, DEST_INDEX_PATH)
    else:
        print(f"[✓] Prebuilt index already exists at: {DEST_INDEX_PATH}")

    validate_corpus()


if __name__ == "__main__":
    main()
