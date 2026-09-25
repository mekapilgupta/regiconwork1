#!/usr/bin/env python3
"""
Generate Multi-View Sentence Bucket & HyDE passages for all Bengali (n=312) and Telugu (n=238) queries.

Generates 7 complete grammatical views per query:
1. trans_literal (Literal translation)
2. trans_natural (Natural encyclopedic translation)
3. trans_alias (English paraphrase with entity aliases)
4. cloze_declarative (Declarative assertion with ____)
5. src_paraphrase (Native source language grammatical paraphrase)
6. src_romanized (Full-sentence Romanized transliteration)
7. hyde_passage (Wikipedia-style opening prose passage with ____)

Includes a strict leakage audit against gold answer spans.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from tqdm.asyncio import tqdm_asyncio

from src.config import DATA_DIR, get_logger
from src.llm_client import call_llm
from src.prompts import render
from src.utils_io import read_jsonl, write_jsonl

log = get_logger("12_gen_multiview")


def audit_view_leakage(views_dict: dict, answers: list[str]) -> tuple[bool, list[str]]:
    leaked = []
    for k in ["cloze_declarative", "hyde_passage"]:
        text = views_dict.get(k, "").lower()
        for ans in answers:
            a_norm = ans.lower().strip()
            # If the answer is a simple common word, skip checking substring false alarms
            if a_norm and len(a_norm) > 3 and a_norm in text:
                leaked.append(f"{k}:{ans}")
    return len(leaked) > 0, leaked


async def process_record(record: dict, client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> dict:
    qid = str(record["id"])
    question = record["question"]
    lang = record["lang"]
    answers = record.get("answers", [])
    out = dict(record)

    p = render("multi_view_v1", lang=lang, question=question)
    res = await call_llm(
        kind="multi_view",
        prompt_version="multi_view_v1",
        input_payload={"lang": lang, "question": question, "id": qid, "task": "multi_view_v1"},
        prompt_text=p,
        client=client,
        semaphore=semaphore,
        require_json=True,
    )

    views = res.parsed_json if isinstance(res.parsed_json, dict) else {}
    is_leak, leak_details = audit_view_leakage(views, answers)
    views["leakage_flag"] = is_leak
    views["leakage_details"] = leak_details

    out["multi_views"] = views
    return out


async def process_track(track: str, semaphore: asyncio.Semaphore):
    file_path = DATA_DIR / "processed" / f"{track}_llm.jsonl"
    records = list(read_jsonl(file_path))
    log.info("Processing multi-views for %d queries from %s...", len(records), file_path)

    batch_size = 25
    processed_all = []
    async with httpx.AsyncClient(timeout=90.0) as client:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            tasks = [process_record(r, client, semaphore) for r in batch]
            results = await tqdm_asyncio.gather(*tasks, desc=f"{track} ({i+1}-{min(i+batch_size, len(records))})")
            processed_all.extend(results)
            await asyncio.sleep(0.2)

    write_jsonl(file_path, processed_all)
    log.info("Saved %d updated queries with multi_views to %s", len(processed_all), file_path)

    leak_count = sum(1 for r in processed_all if r.get("multi_views", {}).get("leakage_flag"))
    log.info("Track %s: Leakage audit detected answer leakage in %d / %d (%.2f%%) queries.",
             track, leak_count, len(processed_all), (leak_count / len(processed_all)) * 100)


async def main():
    semaphore = asyncio.Semaphore(25)
    for track in ["bn", "te"]:
        await process_track(track, semaphore)


if __name__ == "__main__":
    asyncio.run(main())
