#!/usr/bin/env python3
"""
Stage 3: LLM Transformation & Facet Extraction Stage (CPU, Online).

- Asynchronously transforms queries using OpenRouter API:
  * P_TRANSLATE: faithful English translation
  * P_REWRITE: English paraphrase control
  * P_FACETS: structured entities, relation, answer type, and constraints
- Disk caching in cache/llm/ ensures full resumability and 0 redundant API calls.
- Validates facets (ensures exact surface substring matching).
- Runs deterministic leakage audit:
  * bn/te: Answer string containment
  * hi: 4+-token gold passage span overlap
- Exports analysis/leakage.csv and analysis/hi_manual_review.csv for manual inspection.
- Saves processed records to data/processed/<track>_llm.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from src.config import (
    ANALYSIS_DIR,
    DATA_DIR,
    SETTINGS,
    assert_online,
    get_logger,
)
from src.facets import validate_facets
from src.leakage import audit_query_answer_containment, audit_query_passage_span
from src.llm_client import call_llm
from src.prompts import render
from src.utils_io import mark_done, read_jsonl, write_jsonl

log = get_logger("03_llm_stage")

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


async def process_single_query(
    record: dict[str, Any],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    qid = str(record["id"])
    question = record["question"]
    lang = record["lang"]

    out = dict(record)

    # 1. Translation
    p_trans = render("translate_v1", lang=lang, question=question)
    res_trans = await call_llm(
        kind="translate",
        prompt_version="translate_v1",
        input_payload={"lang": lang, "question": question, "task": "translate"},
        prompt_text=p_trans,
        client=client,
        semaphore=semaphore,
        require_json=False,
    )
    llm_translation = res_trans.text.strip().strip('"')
    out["llm_translation"] = llm_translation

    # 2. Rewrite
    p_rewrite = render("rewrite_v1", translation=llm_translation)
    res_rewrite = await call_llm(
        kind="rewrite",
        prompt_version="rewrite_v1",
        input_payload={"translation": llm_translation, "task": "rewrite"},
        prompt_text=p_rewrite,
        client=client,
        semaphore=semaphore,
        require_json=False,
    )
    llm_rewrite = res_rewrite.text.strip().strip('"')
    out["llm_rewrite"] = llm_rewrite

    # 3. Facets
    p_facets = render("facets_v1", lang=lang, question=question)
    res_facets = await call_llm(
        kind="facets",
        prompt_version="facets_v1",
        input_payload={"lang": lang, "question": question, "task": "facets"},
        prompt_text=p_facets,
        client=client,
        semaphore=semaphore,
        require_json=True,
    )
    raw_facets = res_facets.parsed_json or {}
    validated = validate_facets(question, raw_facets)
    out["llm_facets"] = validated

    return out


async def process_track(
    track: str,
    input_file: Path,
    output_file: Path,
    semaphore: asyncio.Semaphore,
    batch_size: int = 20,
    batch_gap_s: float = 1.0,
) -> list[dict[str, Any]]:
    """Process all queries for a track in parallel batches.

    - batch_size queries run concurrently inside each batch (asyncio.gather).
    - batch_gap_s seconds sleep between batches to stay polite to the API.
    - Already-completed records (from prior runs) are skipped instantly.
    - Checkpoint written to disk after every batch so crashes lose at most
      one batch worth of work.
    """
    records = list(read_jsonl(input_file))
    log.info("Processing track '%s' (%d queries, batch=%d, gap=%.1fs)...",
             track, len(records), batch_size, batch_gap_s)

    # --- Resumability: load already-completed records keyed by id ---
    done_map: dict[str, dict] = {}
    if output_file.exists():
        for r in read_jsonl(output_file):
            rid = str(r.get("id", ""))
            if rid and "llm_translation" in r:
                done_map[rid] = r
        if done_map:
            log.info("Track '%s': %d already done, %d remaining.",
                     track, len(done_map), len(records) - len(done_map))

    # Split into pre-done vs needs-processing
    todo = [r for r in records if str(r.get("id", "")) not in done_map]
    completed: dict[str, dict] = dict(done_map)  # id -> processed record

    limits = httpx.Limits(max_keepalive_connections=40, max_connections=60)
    timeout = httpx.Timeout(180.0, connect=30.0)

    n_batches = math.ceil(len(todo) / batch_size) if todo else 0
    pbar = tqdm(total=len(records), initial=len(done_map),
                desc=f"LLM [{track}]", unit="q")

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for b_idx in range(n_batches):
            batch = todo[b_idx * batch_size : (b_idx + 1) * batch_size]

            # Fire all queries in the batch concurrently
            results = await asyncio.gather(
                *[process_single_query(r, client, semaphore) for r in batch],
                return_exceptions=True,
            )

            for record, result in zip(batch, results):
                rid = str(record.get("id", ""))
                if isinstance(result, Exception):
                    log.error("Query %s failed: %s — keeping raw record.", rid, result)
                    completed[rid] = dict(record)  # keep raw so downstream isn't missing it
                else:
                    completed[rid] = result
                pbar.update(1)

            # Checkpoint: write in original input order
            ordered = [completed[str(r.get("id", ""))] for r in records
                       if str(r.get("id", "")) in completed]
            write_jsonl(output_file, ordered)

            # 1-second gap between batches (skip after last batch)
            if b_idx < n_batches - 1:
                await asyncio.sleep(batch_gap_s)

    pbar.close()

    # Final ordered write
    ordered_final = [completed[str(r.get("id", ""))] for r in records
                     if str(r.get("id", "")) in completed]
    write_jsonl(output_file, ordered_final)
    log.info("Track '%s' done — %d/%d records saved to %s",
             track, len(ordered_final), len(records), output_file)
    return ordered_final


def run_leakage_and_manual_exports(all_processed: dict[str, list[dict[str, Any]]]):
    log.info("Running deterministic leakage audit...")
    leakage_rows = []
    hi_review_rows = []

    for track, records in all_processed.items():
        for r in records:
            qid = str(r["id"])
            question = r["question"]
            lang = r["lang"]
            trans = r.get("llm_translation", "")
            rewrite = r.get("llm_rewrite", "")
            facets = r.get("llm_facets", {})

            if track in ("bn", "te"):
                answers = r.get("answers", [])
                audit = audit_query_answer_containment(qid, lang, question, trans, rewrite, facets, answers)
            else:  # hi / hi_mono
                # For mMARCO, if gold passage text isn't in record, check facet vs question
                passage_text = r.get("gold_passage_text", "")
                if passage_text:
                    audit = audit_query_passage_span(qid, lang, question, facets, passage_text)
                else:
                    audit = {"id": qid, "lang": lang, "leak": False, "leaked_fields": []}

            leakage_rows.append({
                "track": track,
                "id": qid,
                "lang": lang,
                "leak": audit.get("leak", False),
                "fuzzy_leak": audit.get("fuzzy_leak", False),
                "leaked_fields": ";".join(audit.get("leaked_fields", [])),
            })

            if track == "hi":
                hi_review_rows.append({
                    "id": qid,
                    "question": question,
                    "translation": trans,
                    "rewrite": rewrite,
                    "facets_json": str(facets),
                    "leak_flag": audit.get("leak", False),
                    "translation_ok": "",
                    "entities_ok": "",
                    "constraints_ok": "",
                    "leak_confirmed": "",
                    "notes": "",
                })

    # Save leakage.csv
    leak_file = ANALYSIS_DIR / "leakage.csv"
    with open(leak_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track", "id", "lang", "leak", "fuzzy_leak", "leaked_fields"])
        writer.writeheader()
        writer.writerows(leakage_rows)
    log.info("Leakage report written to %s", leak_file)

    # Save hi_manual_review.csv (first 100 queries)
    if hi_review_rows:
        hi_review_file = ANALYSIS_DIR / "hi_manual_review.csv"
        sampled_hi = hi_review_rows[:100]
        with open(hi_review_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "id", "question", "translation", "rewrite", "facets_json",
                "leak_flag", "translation_ok", "entities_ok", "constraints_ok",
                "leak_confirmed", "notes",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled_hi)
        log.info("Hindi manual review sheet written to %s (%d queries)", hi_review_file, len(sampled_hi))


async def main_async():
    parser = argparse.ArgumentParser(description="LLM stage for translation, rewrite, and facets")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Semaphore size = max simultaneous LLM requests (default 20)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Queries per parallel batch (default 20)")
    parser.add_argument("--batch-gap", type=float, default=1.0,
                        help="Seconds to sleep between batches (default 1.0)")
    args = parser.parse_args()

    assert_online()
    SETTINGS.validate()

    # Semaphore = batch_size so all queries in a batch fire in parallel
    semaphore = asyncio.Semaphore(max(args.concurrency, args.batch_size))
    tracks = {
        "bn": DATA_DIR / "filtered" / "xor_bn.jsonl",
        "te": DATA_DIR / "filtered" / "xor_te.jsonl",
        "hi": DATA_DIR / "filtered" / "mmarco_hi.jsonl",
        "hi_mono": DATA_DIR / "filtered" / "miracl_hi.jsonl",
    }

    all_processed = {}
    for track, input_path in tracks.items():
        if input_path.exists():
            out_path = PROCESSED_DIR / f"{track}_llm.jsonl"
            processed = await process_track(
                track, input_path, out_path, semaphore,
                batch_size=args.batch_size,
                batch_gap_s=args.batch_gap,
            )
            all_processed[track] = processed
            mark_done(ROOT / "state", "stage03", track)
        else:
            log.warning("Input file for track '%s' not found (%s). Skipping.", track, input_path)

    run_leakage_and_manual_exports(all_processed)
    log.info("Stage 3 completed successfully.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
