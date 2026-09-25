#!/usr/bin/env python
"""RUN ON A NODE WITH INTERNET ACCESS (e.g. the CPU login node), NOT the
offline GPU node. Uses the OpenRouter API to get an ordinal 0/1/2 factuality
verdict for a bounded sample of (source, generated_summary) pairs per
system/language, following the human-eval protocol precedent from
Maynez/FRANK/SummEval/QAGS/FactCC/AggreFact:

  0 = incorrect or hallucinated
  1 = partially correct or incomplete
  2 = factually correct

STRICT COST CONTROL & HIGH CONCURRENCY:
  - Hard cap at configs/pipeline.yaml -> openrouter.max_total_usd (default $10.00)
  - Thread-safe BudgetTracker with pre-call check and real-cost reconciliation
  - Multithreaded concurrent API calls (default 8 workers) with tqdm progress bar
  - Auto-retries on empty/unparseable outputs (up to 3 retries per sample)
  - Resumes automatically from outputs/llm_judge_results.jsonl (re-trying any previous nulls)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import threading
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.budget_tracker import BudgetExceededError, BudgetTracker  # noqa: E402
from utils.io_utils import (  # noqa: E402
    REPO_ROOT,
    get_logger,
    load_config,
    read_jsonl,
    resolve_path,
)
from utils.openrouter_client import OpenRouterClient  # noqa: E402

JUDGE_SYSTEM_PROMPT = """You are a strict, objective news factuality and hallucination judge.
You will be provided with a SOURCE ARTICLE and a GENERATED SUMMARY.
Judge ONLY whether the factual claims in the summary (persons, organizations, locations, events/relations, numbers, dates) are strictly grounded in and supported by the source article. Ignore style, grammar, and fluency.

Respond ONLY in valid JSON with this exact schema:
{
  "verdict": 0 | 1 | 2,
  "flagged_fact_type": "person_identity" | "org_identity" | "location_entity" | "numeric_fact" | "temporal_fact" | "event_relation" | "unsupported_claim" | "omission" | "malformed_output" | "none",
  "reason": "<one concise sentence detailing the factual alignment or specific error>"
}

Verdict Scale:
- 0: Severe contradiction / hallucination (an entity, event, number, or key fact is false, inverted, or fabricated).
- 1: Minor discrepancy / partial omission (core facts present are true, but secondary detail is missing or slightly ambiguous).
- 2: Fully factual (all checkable statements are completely faithful to and grounded in the source)."""


def build_user_prompt(source_text: str, summary_text: str) -> str:
    src_trunc = source_text[:2500]
    return f"SOURCE ARTICLE:\n{src_trunc}\n\nGENERATED SUMMARY:\n{summary_text}\n\nRespond with the JSON verdict now."


def parse_verdict(raw_text: str | None) -> dict:
    if not raw_text:
        return {"verdict": None, "reason": "EMPTY_OR_NONE_RESPONSE", "flagged_fact_type": None}
    try:
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx : end_idx + 1]
        parsed = json.loads(cleaned)
        assert parsed.get("verdict") in (0, 1, 2)
        return parsed
    except Exception:
        preview = str(raw_text)[:200] if raw_text is not None else "None"
        return {"verdict": None, "reason": f"UNPARSEABLE_RESPONSE: {preview}", "flagged_fact_type": None}


def select_sample(consistency_rows: list[dict], summary_rows_by_id: dict, n: int, seed: int) -> list[dict]:
    flagged = [r for r in consistency_rows if r.get("winner_flip_suspected") is True]
    unflagged = [r for r in consistency_rows if r.get("winner_flip_suspected") is not True]

    rng = random.Random(seed)
    rng.shuffle(flagged)
    rng.shuffle(unflagged)

    sample = flagged[:n]
    remaining = n - len(sample)
    if remaining > 0:
        sample += unflagged[:remaining]

    out = []
    for r in sample:
        src = summary_rows_by_id.get(r["id"])
        if src is None:
            continue
        out.append(
            {
                "id": r["id"],
                "system": r["system"],
                "language": r["language"],
                "source_text": src["source_text"],
                "generated_summary": src["generated_summary"],
                "winner_flip_suspected": r.get("winner_flip_suspected"),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent worker threads")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite previous judge results")
    parser.add_argument("--dry_run", action="store_true", help="Select sample and print cost projection, no API calls")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("llm_judge", cfg)
    ocfg = cfg["openrouter"]

    tracker = BudgetTracker(
        state_path=REPO_ROOT / cfg["paths"]["outputs_dir"] / "budget_state.json",
        max_total_usd=ocfg["max_total_usd"],
        safety_margin_usd=ocfg["safety_margin_usd"],
    )
    logger.info(f"Budget tracker initialized. {tracker.summary()}")

    n_per = ocfg["samples_per_language_per_system"]

    # Build the full sample plan across all system/language combos
    plan: list[dict] = []
    for consistency_file in sorted((REPO_ROOT / cfg["paths"]["outputs_dir"] / "fact_consistency").glob("*.jsonl")):
        fname = consistency_file.stem.replace("__consistency", "")
        summary_file = REPO_ROOT / cfg["paths"]["outputs_dir"] / "summaries" / f"{fname}.jsonl"
        if not summary_file.exists():
            logger.warning(f"Missing summary file for {fname}, skipping.")
            continue

        consistency_rows = list(read_jsonl(consistency_file))
        summary_rows = {r["id"]: r for r in read_jsonl(summary_file)}
        sample = select_sample(consistency_rows, summary_rows, n_per, cfg["xlsum"]["random_seed"])
        plan.extend(sample)

    logger.info(f"Planned {len(plan)} total LLM-judge calls across all systems/languages.")

    projected_worst_case = (
        len(plan) * (ocfg["max_output_tokens_per_call"] / 1000.0) * tracker.conservative_price_per_1k_output_tokens
    )
    logger.info(f"Projected worst-case cost for full plan: ${projected_worst_case:.4f} (cap: ${ocfg['max_total_usd']:.2f})")

    if args.dry_run:
        logger.info("Dry run complete, no API calls made.")
        return

    client = OpenRouterClient(
        base_url=ocfg["base_url"],
        model_id=ocfg["judge_model_id"],
        api_key_env_var=ocfg["api_key_env_var"],
        tracker=tracker,
        poll_generation_cost=ocfg["poll_generation_cost"],
    )

    out_path = resolve_path(cfg, "outputs_dir", "llm_judge_results.jsonl")
    if args.overwrite and out_path.exists():
        out_path.unlink()
        logger.info(f"Overwriting previous judge file -> {out_path}")
    
    # Read existing valid results
    valid_existing_records = []
    already_done_keys = set()
    if out_path.exists():
        for r in read_jsonl(out_path):
            # Only count as already done if verdict is non-null (0, 1, or 2)
            if r.get("verdict") in (0, 1, 2):
                valid_existing_records.append(r)
                already_done_keys.add((r["system"], r["language"], r["id"]))
        
        # Clean the file by writing back only valid records
        with open(out_path, "w", encoding="utf-8") as f:
            for r in valid_existing_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Resuming: {len(already_done_keys)} valid examples already judged. Cleaned any null/failed entries.")

    pending = [item for item in plan if (item["system"], item["language"], item["id"]) not in already_done_keys]
    logger.info(f"Dispatching {len(pending)} items across {args.workers} worker threads using {ocfg['judge_model_id']}...")

    file_lock = threading.Lock()
    stop_event = threading.Event()
    n_completed = 0

    def judge_worker(item: dict) -> bool:
        if stop_event.is_set():
            return False

        key = (item["system"], item["language"], item["id"])
        user_prompt = build_user_prompt(item["source_text"], item["generated_summary"])
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        verdict_dict = {"verdict": None, "reason": None, "flagged_fact_type": None}
        cost = 0.0

        # Try up to 3 times to obtain a valid parsed verdict
        for attempt in range(3):
            if stop_event.is_set():
                return False
            try:
                result = client.chat_completion(
                    messages=messages,
                    max_output_tokens=ocfg["max_output_tokens_per_call"],
                    temperature=ocfg["temperature"],
                    call_id=f"{item['system']}__{item['language']}__{item['id']}_att{attempt}",
                    meta={"system": item["system"], "language": item["language"], "id": item["id"]},
                )
                cost += result.get("cost_usd", 0.0)
                parsed = parse_verdict(result.get("text"))
                if parsed["verdict"] in (0, 1, 2):
                    verdict_dict = parsed
                    break
            except BudgetExceededError as e:
                logger.error(f"BUDGET CAP REACHED: {e}")
                stop_event.set()
                return False
            except Exception as e:
                logger.warning(f"[{attempt+1}/3] Call failed for {key}: {e}")

        if verdict_dict["verdict"] is None:
            logger.error(f"Failed to obtain valid verdict for {key} after retries.")

        record = {
            "system": item["system"],
            "language": item["language"],
            "id": item["id"],
            "winner_flip_suspected": item["winner_flip_suspected"],
            "verdict": verdict_dict["verdict"],
            "reason": verdict_dict.get("reason"),
            "flagged_fact_type": verdict_dict.get("flagged_fact_type"),
            "cost_usd": cost,
        }

        with file_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

        return True

    with tqdm(total=len(pending), desc="LLM Judging Progress") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(judge_worker, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                try:
                    ok = future.result()
                    if ok:
                        n_completed += 1
                except Exception as e:
                    logger.error(f"Worker exception: {e}")
                pbar.update(1)
                pbar.set_postfix({"Spent": f"${tracker.spent_usd:.4f}", "Remaining": f"${tracker.remaining_usd:.2f}"})

    logger.info(f"Done. Successfully judged {n_completed} examples this run. {tracker.summary()}")
    logger.info(f"Results saved to -> {out_path}")


if __name__ == "__main__":
    main()
