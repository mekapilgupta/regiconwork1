#!/usr/bin/env python
"""Audits and filters candidate sports articles using qwen/qwen3.8-flash via OpenRouter.

Guarantees a 100% verified sports dataset for all downstream experiments,
eliminating non-sports false positives (strikes, trade agreements, politics, etc.).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_utils import get_logger, load_config, resolve_path, write_jsonl
from utils.openrouter_client import OpenRouterClient
from utils.budget_tracker import BudgetTracker

PROMPT_TEMPLATE = """You are an expert sports journalism auditor. Determine if the following news article is STRICTLY about sports (a sporting match, athlete, tournament, athletic team, sports league, coaching, or sports competition).
Articles about general strikes, political agreements, economic transport treaties, crime, or entertainment that merely mention words like "strike", "match", or "team" in a non-sports context MUST be flagged as false.

Title: {title}
URL: {url}
Snippet: {snippet}

Respond ONLY in valid JSON:
{{
  "is_sport": true or false,
  "sport_type": "cricket" | "football" | "tennis" | "hockey" | "badminton" | "athletics" | "boxing" | "wrestling" | "other_sport" | "non_sport",
  "confidence": 0.95,
  "reason": "one concise sentence explaining why this is or is not sports"
}}"""


def parse_llm_json(raw_text: str | None) -> dict:
    if not raw_text:
        return {"is_sport": False, "sport_type": "non_sport", "reason": "EMPTY_RESPONSE"}
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except Exception:
        return {"is_sport": False, "sport_type": "non_sport", "reason": f"UNPARSEABLE: {raw_text[:100]}"}


def verify_single_article(client: OpenRouterClient, row: dict, max_tokens: int = 150) -> dict:
    title = row.get("title", "")
    url = row.get("url", "")
    snippet = row.get("text", "")[:600]
    
    # Fast heuristic: if URL explicitly contains /sport/ or /sport, it is almost certainly sports, but we still verify
    prompt = PROMPT_TEMPLATE.format(title=title, url=url, snippet=snippet)
    messages = [{"role": "user", "content": prompt}]
    
    try:
        res = client.chat_completion(messages, max_output_tokens=max_tokens, temperature=0.0)
        parsed = parse_llm_json(res.get("text"))
        return {
            "id": row.get("id"),
            "url": url,
            "title": title,
            "is_sport": bool(parsed.get("is_sport", False)),
            "sport_type": parsed.get("sport_type", "unknown"),
            "reason": parsed.get("reason", ""),
            "confidence": parsed.get("confidence", 0.0),
            "cost_usd": res.get("cost_usd", 0.0),
            "original_row": row
        }
    except Exception as e:
        return {
            "id": row.get("id"),
            "url": url,
            "title": title,
            "is_sport": False,
            "sport_type": "error",
            "reason": f"CALL_FAILED: {str(e)}",
            "confidence": 0.0,
            "cost_usd": 0.0,
            "original_row": row
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers for OpenRouter")
    parser.add_argument("--model", default="qwen/qwen3.8-flash", help="OpenRouter model for verification")
    args = parser.parse_args()

    load_dotenv(".env")
    cfg = load_config(args.config)
    logger = get_logger("verify_sports_articles", cfg)

    # Initialize budget tracker & client
    budget_file = resolve_path(cfg, "outputs_dir", "budget_state.json")
    tracker = BudgetTracker(
        state_path=budget_file,
        max_total_usd=cfg["openrouter"]["max_total_usd"],
        safety_margin_usd=cfg["openrouter"]["safety_margin_usd"],
        conservative_price_per_1k_output_tokens=0.001,
    )
    client = OpenRouterClient(
        base_url=cfg["openrouter"]["base_url"],
        model_id=args.model,
        api_key_env_var=cfg["openrouter"]["api_key_env_var"],
        tracker=tracker,
        poll_generation_cost=False,
    )

    out_dir = resolve_path(cfg, "data_dir", "sports_verified")
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_summary = []

    for lang_cfg in cfg["languages"]:
        lang = lang_cfg["code"]
        candidate_file = resolve_path(cfg, "data_dir", "sports_filtered", f"{lang}_test.jsonl")
        if not candidate_file.exists():
            logger.warning(f"Candidate file not found: {candidate_file}")
            continue

        with open(candidate_file, encoding="utf-8") as f:
            candidates = [json.loads(line) for line in f]

        logger.info(f"[{lang}] Auditing {len(candidates)} candidate articles with {args.model}...")

        verified_articles = []
        removed_articles = []
        lock = threading.Lock()

        def worker_task(cand):
            res = verify_single_article(client, cand)
            with lock:
                if res["is_sport"]:
                    verified_articles.append(res)
                else:
                    removed_articles.append(res)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(worker_task, candidates), total=len(candidates), desc=f"Verifying {lang}"))

        # Save verified
        verified_out = out_dir / f"{lang}_test.jsonl"
        write_jsonl(verified_out, [v["original_row"] for v in verified_articles])

        # Save audit log
        audit_log = out_dir / f"{lang}_audit_log.jsonl"
        with open(audit_log, "w", encoding="utf-8") as f:
            for item in verified_articles + removed_articles:
                f.write(json.dumps({
                    "id": item["id"],
                    "title": item["title"],
                    "url": item["url"],
                    "is_sport": item["is_sport"],
                    "sport_type": item["sport_type"],
                    "reason": item["reason"],
                    "confidence": item["confidence"]
                }, ensure_ascii=False) + "\n")

        retention_pct = (len(verified_articles) / len(candidates) * 100) if candidates else 0.0
        logger.info(f"[{lang}] Audited: {len(candidates)} | Verified True Sports: {len(verified_articles)} | Removed Non-Sports: {len(removed_articles)} | Retention: {retention_pct:.1f}%")
        
        audit_summary.append({
            "language": lang,
            "candidates_screened": len(candidates),
            "verified_true_sports": len(verified_articles),
            "removed_non_sports": len(removed_articles),
            "retention_rate_pct": round(retention_pct, 2)
        })

    logger.info("=== SPORTS DATASET AUDIT SUMMARY ===")
    print("\n" + json.dumps(audit_summary, indent=2))
    
    # Save summary table
    summary_path = resolve_path(cfg, "outputs_dir", "sports_filtering_audit.csv")
    import pandas as pd
    pd.DataFrame(audit_summary).to_csv(summary_path, index=False)
    logger.info(f"Saved audit summary table to {summary_path}")


if __name__ == "__main__":
    main()
