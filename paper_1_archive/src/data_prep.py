"""
Turns raw downloaded files into per-track query lists, ready for the LLM
stage. Everything here is cheap/CPU-light and safe to run on the online
CPU node.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR, get_logger
from src.utils_io import read_jsonl, write_jsonl, nfc

log = get_logger("data_prep")

XOR_LANGS = {"bn", "te"}
SEED = 42


def filter_xor_tydi_bn_te(raw_path: Path, n: Optional[int] = 100) -> dict[str, Path]:
    """Filter xor_dev_retrieve_eng_span.jsonl to lang in {bn, te}. Keeps
    fields: id, question, lang, answers, eng_translation (human).
    If n is given, deterministically subsamples n per language (seed=42);
    n=None or n<0 keeps everything."""
    by_lang: dict[str, list[dict]] = {"bn": [], "te": []}
    for rec in read_jsonl(raw_path):
        lang = rec.get("lang")
        if lang not in XOR_LANGS:
            continue
        item = {
            "id": rec.get("id"),
            "question": nfc(rec.get("question", "")),
            "lang": lang,
            "answers": [nfc(a) for a in rec.get("answers", [])],
            "eng_translation": nfc(rec.get("eng_translation") or rec.get("translation") or ""),
        }
        by_lang[lang].append(item)

    out_paths = {}
    for lang, items in by_lang.items():
        items = sorted(items, key=lambda r: str(r["id"]))  # deterministic order before sampling
        if n is not None and n >= 0 and n < len(items):
            rnd = random.Random(SEED)
            items = rnd.sample(items, n)
            items = sorted(items, key=lambda r: str(r["id"]))
        out_path = DATA_DIR / "filtered" / f"xor_{lang}.jsonl"
        write_jsonl(out_path, items)
        log.info("xor_tydi[%s]: %d queries -> %s", lang, len(items), out_path)
        out_paths[lang] = out_path
    return out_paths


def sample_mmarco_hi(
    queries_tsv: Path,
    qrels_tsv: Path,
    en_queries_tsv: Optional[Path] = None,
    n: Optional[int] = 100,
    full_target_size: int = 1000,
) -> Path:
    """Spec §1b: sample 1000 (or n for a pilot) from the 6,980 dev.small hi
    queries with a fixed seed=42, deterministic and reproducible. Attaches
    the MS MARCO qrels (language-independent) query->passage id mapping,
    and original English MS MARCO queries for the S8 Oracle baseline."""
    qrels: dict[str, list[str]] = {}
    with open(qrels_tsv, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4:
                continue
            qid, _, pid, rel = row
            if int(rel) > 0:
                qrels.setdefault(qid, []).append(pid)

    en_queries: dict[str, str] = {}
    if en_queries_tsv and en_queries_tsv.exists():
        with open(en_queries_tsv, "r", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 2:
                    en_queries[row[0]] = nfc(row[1])

    rows = []
    with open(queries_tsv, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            qid, qtext = row[0], row[1]
            if qid in qrels:  # only keep queries that have relevance judgments
                rows.append({
                    "id": qid,
                    "question": nfc(qtext),
                    "lang": "hi",
                    "relevant_passage_ids": qrels[qid],
                    "eng_translation": en_queries.get(qid, ""),  # Oracle English MS MARCO query
                })

    rows = sorted(rows, key=lambda r: str(r["id"]))
    target = n if (n is not None and n >= 0) else full_target_size
    rnd = random.Random(SEED)
    sample_size = min(target, len(rows))
    sampled = rnd.sample(rows, sample_size)
    sampled = sorted(sampled, key=lambda r: str(r["id"]))

    out_path = DATA_DIR / "filtered" / "mmarco_hi.jsonl"
    write_jsonl(out_path, sampled)
    log.info(
        "mmarco_hi: sampled %d/%d queries (seed=%d) -> %s",
        len(sampled),
        len(rows),
        SEED,
        out_path,
    )
    return out_path


def load_miracl_hi_monolingual(topics_tsv: Path, qrels_tsv: Path, n: Optional[int] = 100) -> Path:
    """Spec §1b: Monolingual control MIRACL hi dev (500 queries or n for pilot)."""
    qrels: dict[str, list[str]] = {}
    if qrels_tsv.exists():
        with open(qrels_tsv, "r", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) < 4:
                    continue
                qid, _, pid, rel = row
                if int(rel) > 0:
                    qrels.setdefault(qid, []).append(pid)

    rows = []
    with open(topics_tsv, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            qid, qtext = row[0], row[1]
            rows.append({
                "id": qid,
                "question": nfc(qtext),
                "lang": "hi",
                "relevant_passage_ids": qrels.get(qid, []),
            })

    rows = sorted(rows, key=lambda r: str(r["id"]))
    target = n if (n is not None and n >= 0) else 500
    rnd = random.Random(SEED)
    sample_size = min(target, len(rows))
    sampled = rnd.sample(rows, sample_size)
    sampled = sorted(sampled, key=lambda r: str(r["id"]))

    out_path = DATA_DIR / "filtered" / "miracl_hi.jsonl"
    write_jsonl(out_path, sampled)
    log.info("miracl_hi: sampled %d queries -> %s", len(sampled), out_path)
    return out_path
