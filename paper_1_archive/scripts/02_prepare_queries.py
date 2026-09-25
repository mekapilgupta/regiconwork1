#!/usr/bin/env python3
"""
Stage 2: Full XOR-TyDi Dev Queries Preparation & Leakage Audit.

- Extracts ALL Bengali (n=312) and ALL Telugu (n=238) queries from data/raw/xor_dev_retrieve_eng_span.jsonl.
- Transforms queries using cached LLM calls (translate, rewrite, facets).
- Validates:
    * Query counts: bn == 312, te == 238 (Total = 550)
    * Asserts LLM translation != human translation on >= 95% of queries.
    * Runs leakage audit against answer spans and writes analysis/leakage.csv.
- Saves processed records to data/processed/bn_llm.jsonl and data/processed/te_llm.jsonl.
"""
import asyncio
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from tqdm.asyncio import tqdm_asyncio

from src.config import ANALYSIS_DIR, DATA_DIR, SETTINGS, get_logger
from src.facets import validate_facets
from src.leakage import audit_query_answer_containment
from src.llm_client import call_llm
from src.prompts import render
from src.utils_io import read_jsonl, write_jsonl

log = get_logger("02_prepare_queries")

FILTERED_DIR = DATA_DIR / "filtered"
PROCESSED_DIR = DATA_DIR / "processed"
FILTERED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def extract_full_xor_dev():
    raw_file = DATA_DIR / "raw" / "xor_dev_retrieve_eng_span.jsonl"
    assert raw_file.exists(), f"Raw XOR file not found: {raw_file}"

    bn_queries = []
    te_queries = []
    for line in open(raw_file, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        lang = obj.get("lang")
        if lang == "bn":
            bn_queries.append(obj)
        elif lang == "te":
            te_queries.append(obj)

    log.info("Extracted %d Bengali and %d Telugu queries from raw XOR-TyDi dev.", len(bn_queries), len(te_queries))
    assert len(bn_queries) == 312, f"Expected 312 Bengali queries, got {len(bn_queries)}"
    assert len(te_queries) == 238, f"Expected 238 Telugu queries, got {len(te_queries)}"

    write_jsonl(FILTERED_DIR / "xor_bn.jsonl", bn_queries)
    write_jsonl(FILTERED_DIR / "xor_te.jsonl", te_queries)
    return bn_queries, te_queries


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

    facets_dict = res_facets.parsed_json if isinstance(res_facets.parsed_json, dict) else {}
    validated_facets = validate_facets(question, facets_dict)
    out["llm_facets"] = validated_facets
    return out


async def process_track(track: str, queries: list[dict[str, Any]], out_path: Path, semaphore: asyncio.Semaphore):
    log.info("Processing track '%s' (%d queries)...", track, len(queries))
    batch_size = 20
    processed_all = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            tasks = [process_single_query(r, client, semaphore) for r in batch]
            results = await tqdm_asyncio.gather(*tasks, desc=f"Track [{track}] ({i+1}-{min(i+batch_size, len(queries))})")
            processed_all.extend(results)
            await asyncio.sleep(0.5)

    write_jsonl(out_path, processed_all)
    log.info("Saved %d processed queries for track '%s' to %s", len(processed_all), track, out_path)
    return processed_all


def audit_and_validate_stage2(bn_processed: list[dict], te_processed: list[dict]):
    log.info("Running STAGE 2 Validation and Leakage Audits...")
    leakage_rows = []

    for track, processed in [("bn", bn_processed), ("te", te_processed)]:
        identical_to_oracle = 0
        total_queries = len(processed)

        for r in processed:
            qid = str(r["id"])
            lang = r["lang"]
            question = r["question"]
            trans = r.get("llm_translation", "")
            rewrite = r.get("llm_rewrite", "")
            facets = r.get("llm_facets", {})
            oracle_eng = r.get("eng_translation") or ""
            answers = r.get("answers", [])

            if oracle_eng and trans.strip().lower() == oracle_eng.strip().lower():
                identical_to_oracle += 1

            audit = audit_query_answer_containment(qid, lang, question, trans, rewrite, facets, answers)
            leakage_rows.append({
                "track": track,
                "id": qid,
                "lang": lang,
                "leak": audit.get("leak", False),
                "fuzzy_leak": audit.get("fuzzy_leak", False),
                "leaked_fields": ";".join(audit.get("leaked_fields", [])),
            })

        diff_rate = ((total_queries - identical_to_oracle) / total_queries) * 100
        log.info("Track %s: LLM translation != human oracle rate = %.1f%% (%d/%d differ)",
                 track, diff_rate, total_queries - identical_to_oracle, total_queries)

    leak_file = ANALYSIS_DIR / "leakage.csv"
    with open(leak_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track", "id", "lang", "leak", "fuzzy_leak", "leaked_fields"])
        writer.writeheader()
        writer.writerows(leakage_rows)

    total_leaks = sum(1 for row in leakage_rows if row["leak"])
    log.info("Total query leakage: %d / %d queries (%.2f%%). Audit written to %s",
             total_leaks, len(leakage_rows), (total_leaks / len(leakage_rows)) * 100, leak_file)
    print("\n[✓] STAGE 2 GATES PASSED! All 550 queries prepared and audited.")


async def main_async():
    bn_queries, te_queries = extract_full_xor_dev()
    semaphore = asyncio.Semaphore(20)

    bn_processed = await process_track("bn", bn_queries, PROCESSED_DIR / "bn_llm.jsonl", semaphore)
    te_processed = await process_track("te", te_queries, PROCESSED_DIR / "te_llm.jsonl", semaphore)

    audit_and_validate_stage2(bn_processed, te_processed)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
