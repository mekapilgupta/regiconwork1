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
        items = sorted(items, key=lambda r: r["id"])  # deterministic order before sampling
        if n is not None and n >= 0 and n < len(items):
            rnd = random.Random(SEED)
            items = rnd.sample(items, n)
            items = sorted(items, key=lambda r: r["id"])
        out_path = DATA_DIR / "filtered" / f"xor_{lang}.jsonl"
        write_jsonl(out_path, items)
        log.info("xor_tydi[%s]: %d queries -> %s", lang, len(items), out_path)
        out_paths[lang] = out_path
    return out_paths


def sample_mmarco_hi(queries_tsv: Path, qrels_tsv: Path, n: Optional[int] = 100,
                      full_pool_size: int = 6980) -> Path:
    """Spec §1b: sample 1000 (or n for a pilot) from the 6,980 dev.small hi
    queries with a fixed seed=42, deterministic and reproducible. Attaches
    the MS MARCO qrels (language-independent) query->passage id mapping."""
    qrels: dict[str, list[str]] = {}
    with open(qrels_tsv, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4:
                continue
            qid, _, pid, rel = row
            if int(rel) > 0:
                qrels.setdefault(qid, []).append(pid)

    rows = []
    with open(queries_tsv, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            qid, qtext = row[0], row[1]
            if qid in qrels:  # only keep queries that have relevance judgments
                rows.append({"id": qid, "question": nfc(qtext), "lang": "hi",
                             "relevant_passage_ids": qrels[qid]})

    rows = sorted(rows, key=lambda r: r["id"])
    target = n if (n is not None and n >= 0) else 1000
    rnd = random.Random(SEED)
    sample_size = min(target, len(rows))
    sampled = rnd.sample(rows, sample_size)
    sampled = sorted(sampled, key=lambda r: r["id"])

    out_path = DATA_DIR / "filtered" / "mmarco_hi.jsonl"
    write_jsonl(out_path, sampled)
    log.info("mmarco_hi: sampled %d/%d queries (seed=%d) -> %s",
              len(sampled), len(rows), SEED, out_path)
    return out_path


def load_mrtydi_or_miracl_monolingual(lang: str, n: Optional[int] = 100) -> Path:
    """Loads the Mr.TyDi/MIRACL monolingual `lang` dev split via the
    `ir_datasets` package (which pyserini also depends on / can drive),
    as the monolingual control track. Falls back to a clear error telling
    the operator to run scripts/02_download_cpu_online.py --monolingual
    first if the dataset isn't cached locally.
    """
    import ir_datasets  # lazy import; only needed for this track

    dataset_name = f"mrtydi/{lang}/dev" if lang in ("bn", "te") else f"miracl/{lang}/dev"
    log.info("Loading monolingual control dataset: %s", dataset_name)
    ds = ir_datasets.load(dataset_name)

    rows = []
    for q in ds.queries_iter():
        rows.append({"id": q.query_id, "question": nfc(q.text), "lang": lang})
    rows = sorted(rows, key=lambda r: r["id"])

    target = n if (n is not None and n >= 0) else (500 if lang == "hi" else len(rows))
    rnd = random.Random(SEED)
    sample_size = min(target, len(rows))
    sampled = rnd.sample(rows, sample_size)
    sampled = sorted(sampled, key=lambda r: r["id"])

    out_path = DATA_DIR / "filtered" / f"mono_{lang}.jsonl"
    write_jsonl(out_path, sampled)
    log.info("monolingual[%s]: %d queries -> %s", lang, len(sampled), out_path)
    return out_path
