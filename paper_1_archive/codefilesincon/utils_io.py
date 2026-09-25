"""
IO + resumability helpers shared by every stage.

- All text I/O is explicit utf-8. Bengali/Telugu/Hindi strings are passed
  through unicodedata.normalize("NFC", ...) on read, so downstream substring
  checks (e.g. facet surface_form validation) aren't broken by combining-mark
  order differences between sources.
- JSONL read/write with orjson if available (fast), falling back to stdlib json.
- sha256 hashing for the LLM cache key.
- Simple "stage done" marker files for resumability at the pipeline-stage
  granularity, on top of the per-item cache-file resumability that already
  covers LLM calls and embeddings.
"""
from __future__ import annotations

import hashlib
import json as _json
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

try:
    import orjson

    def _dumps(obj: Any) -> str:
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")

    def _loads(s: str) -> Any:
        return orjson.loads(s)
except ImportError:  # pragma: no cover
    def _dumps(obj: Any) -> str:
        return _json.dumps(obj, ensure_ascii=False)

    def _loads(s: str) -> Any:
        return _json.loads(s)


def nfc(text: Optional[str]) -> Optional[str]:
    """Normalize Bengali/Telugu/Hindi (or any) text to NFC. Idempotent."""
    if text is None:
        return None
    return unicodedata.normalize("NFC", text)


def normalize_record_strings(rec: dict) -> dict:
    """Recursively NFC-normalize every string value in a dict/list, leaving
    other types untouched. Used right after reading any raw dataset file."""
    if isinstance(rec, dict):
        return {k: normalize_record_strings(v) for k, v in rec.items()}
    if isinstance(rec, list):
        return [normalize_record_strings(v) for v in rec]
    if isinstance(rec, str):
        return nfc(rec)
    return rec


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield normalize_record_strings(_loads(line))


def write_jsonl(path: Path, records: Iterable[dict], mode: str = "w") -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(_dumps(rec))
            f.write("\n")
            n += 1
    return n


def append_jsonl_dedup(path: Path, record: dict, id_field: str = "id") -> None:
    """Append a record unless an id-matching record is already present.
    Used for slow, resumable append-only files (e.g. judge labels)."""
    existing_ids = set()
    if path.exists():
        for rec in read_jsonl(path):
            existing_ids.add(rec.get(id_field))
    if record.get(id_field) in existing_ids:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(_dumps(record))
        f.write("\n")


def sha256_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Stage resumability markers
# ---------------------------------------------------------------------------

def mark_done(state_dir: Path, stage: str, track: str, payload: Optional[dict] = None) -> None:
    p = state_dir / f"{stage}__{track}.done"
    p.write_text(_dumps(payload or {"stage": stage, "track": track}), encoding="utf-8")


def is_done(state_dir: Path, stage: str, track: str) -> bool:
    return (state_dir / f"{stage}__{track}.done").exists()


def require_done(state_dir: Path, stage: str, track: str) -> None:
    if not is_done(state_dir, stage, track):
        raise RuntimeError(
            f"Prerequisite stage '{stage}' has not completed for track '{track}'. "
            f"Run the corresponding script first."
        )


# ---------------------------------------------------------------------------
# Bounded thread-pool map (for I/O-bound, CPU-light work: BM25 lookups,
# file parsing, leakage checks). Heavy tensor work should NOT use this —
# batch it on the GPU instead (see src/embeddings.py).
# ---------------------------------------------------------------------------

T = TypeVar("T")
R = TypeVar("R")


def threaded_map(fn: Callable[[T], R], items: list[T], max_workers: int = 8,
                  desc: str = "") -> list[R]:
    from tqdm import tqdm

    results: list[Optional[R]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i = futures[fut]
            results[i] = fut.result()
    return results  # type: ignore
