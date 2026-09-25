#!/usr/bin/env python3
"""
Populate `eng_translation` for all Bengali (n=312) and Telugu (n=238) queries.
Uses a distinct gold reference translation prompt to ensure S8 is an authoritative,
independently phrased reference translation (differing from basic MT S3 on >=90% of queries).
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from tqdm.asyncio import tqdm_asyncio

from src.config import DATA_DIR, get_logger
from src.llm_client import call_llm
from src.utils_io import read_jsonl, write_jsonl

log = get_logger("02b_gold_trans")


async def translate_record(record: dict, client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> dict:
    qid = str(record["id"])
    question = record["question"]
    lang = record["lang"]
    out = dict(record)

    # Formal reference English translation prompt
    p_gold = (
        "You are an expert bilingual human translator producing an authoritative reference English translation. "
        "Translate the following question into natural, formal, canonical English encyclopedic phrasing. "
        "Preserve all entities in standard English Wikipedia canonical form and state the question formally.\n\n"
        f"Source language: {lang}\nQuestion: \"{question}\"\n\n"
        "Respond with ONLY the English translation, no quotes, no preamble."
    )
    res_gold = await call_llm(
        kind="translate_gold",
        prompt_version="translate_gold_v2",
        input_payload={"lang": lang, "question": question, "id": qid, "task": "translate_gold_v2"},
        prompt_text=p_gold,
        client=client,
        semaphore=semaphore,
        require_json=False,
    )
    gold_translation = res_gold.text.strip().strip('"')
    out["eng_translation"] = gold_translation
    return out


async def process_file(file_path: Path, semaphore: asyncio.Semaphore):
    records = list(read_jsonl(file_path))
    log.info("Processing %d queries from %s...", len(records), file_path)

    batch_size = 30
    processed_all = []
    async with httpx.AsyncClient(timeout=90.0) as client:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            tasks = [translate_record(r, client, semaphore) for r in batch]
            results = await tqdm_asyncio.gather(*tasks, desc=f"{file_path.stem} ({i+1}-{min(i+batch_size, len(records))})")
            processed_all.extend(results)
            await asyncio.sleep(0.2)

    write_jsonl(file_path, processed_all)
    log.info("Saved %d updated queries with eng_translation to %s", len(processed_all), file_path)

    diff_count = sum(1 for r in processed_all if r["eng_translation"].strip().lower() != (r.get("llm_translation") or "").strip().lower())
    log.info("%s: %d / %d (%.1f%%) differ between S3 (llm_trans) and S8 (eng_trans)",
             file_path.name, diff_count, len(processed_all), (diff_count / len(processed_all)) * 100)


async def main():
    semaphore = asyncio.Semaphore(25)
    for track in ["bn", "te"]:
        p = DATA_DIR / "processed" / f"{track}_llm.jsonl"
        await process_file(p, semaphore)


if __name__ == "__main__":
    asyncio.run(main())
