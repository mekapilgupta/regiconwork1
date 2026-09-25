#!/usr/bin/env python
"""Builds a verified sports benchmark dataset across English, Hindi, and Bengali.
Extracts top sports candidates from the full dataset and verifies each with qwen/qwen3.8-flash.
Target: ~40-50 verified sports articles per language.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import sys
import threading
from pathlib import Path

from datasets import load_from_disk
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.budget_tracker import BudgetTracker
from utils.io_utils import get_logger, load_config, resolve_path, write_jsonl
from utils.openrouter_client import OpenRouterClient
import importlib

verify_module = importlib.import_module("scripts.02b_verify_sports_articles")
verify_single_article = verify_module.verify_single_article

PATTERNS = {
    "english": re.compile(
        r"\b(cricket|football|soccer|tennis|olympic|athletics|championship|premier league|fifa|world cup|batsman|bowler|wicket|striker|wimbledon|badminton|boxing|formula 1|f1|nba|ipl|liverpool|chelsea|arsenal|manchester|messi|ronaldo)\b",
        re.I,
    ),
    "hindi": re.compile(
        r"(खेल|क्रिकेट|फ़ुटबॉल|टेनिस|हॉकी|ओलंपिक|विश्वकप|टूर्नामेंट|आईपीएल|बल्लेबाज़|गेंदबाज़|विकेट|खिलाड़ी|गोल्ड मेडल|विराट|कोहली|धोनी|रोहित|मेसी|रोनाल्डो)",
        re.I,
    ),
    "bengali": re.compile(
        r"(খেলা|ক্রিকেট|ফুটবল|টেনিস|হকি|অলিম্পিক|বিশ্বকাপ|টুর্নামেন্ট|আইপিএল|ব্যাটসম্যান|বোলার|উইকেট|খেলোয়াড়|গোল্ড মেডেল|সাকিব|তামিম|মাশরাফি|মেসি|রোনালদো|কোহলি|মুশফিক|টাইগার)",
        re.I,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--target_per_lang", type=int, default=40, help="Target verified sports articles per language")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    load_dotenv(".env")
    cfg = load_config(args.config)
    logger = get_logger("build_verified_benchmark", cfg)

    tracker = BudgetTracker(
        state_path=resolve_path(cfg, "outputs_dir", "budget_state.json"),
        max_total_usd=cfg["openrouter"]["max_total_usd"],
        safety_margin_usd=cfg["openrouter"]["safety_margin_usd"],
        conservative_price_per_1k_output_tokens=0.001,
    )
    client = OpenRouterClient(
        base_url=cfg["openrouter"]["base_url"],
        model_id="qwen/qwen3.8-flash",
        api_key_env_var=cfg["openrouter"]["api_key_env_var"],
        tracker=tracker,
        poll_generation_cost=False,
    )

    out_dir = resolve_path(cfg, "data_dir", "sports_verified")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    final_summary = []

    for lang in cfg["languages"]:
        code = lang["code"]
        raw_dir = resolve_path(cfg, "data_dir", "xlsum_raw", code)
        if not raw_dir.exists():
            continue

        ds = load_from_disk(str(raw_dir))
        pat = PATTERNS[code]
        all_matches = []

        for split_name in ["test", "validation", "train"]:
            if split_name not in ds:
                continue
            for row in ds[split_name]:
                title = row.get("title", "")
                url = row.get("url", "")
                if "/sport" in url or "/khel" in url or "/khela" in url or pat.search(title):
                    all_matches.append({**row, "split_source": split_name})

        # Deduplicate
        unique_matches = {r["id"]: r for r in all_matches}
        match_pool = list(unique_matches.values())
        rng.shuffle(match_pool)
        logger.info(f"[{code}] Found {len(match_pool)} candidate matches. Starting verification to find {args.target_per_lang} true sports articles...")

        verified = []
        screened_count = 0
        lock = threading.Lock()

        # Stream verification until target reached or pool exhausted
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            batch_size = args.workers * 2
            for i in range(0, len(match_pool), batch_size):
                if len(verified) >= args.target_per_lang:
                    break
                batch = match_pool[i : i + batch_size]
                futures = [executor.submit(verify_single_article, client, row) for row in batch]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    screened_count += 1
                    if res.get("is_sport") is True:
                        with lock:
                            if len(verified) < args.target_per_lang:
                                verified.append(res["original_row"])
                                logger.info(f"[{code}] Verified sports article [{len(verified)}/{args.target_per_lang}]: {res['title'][:60]}")

        out_file = out_dir / f"{code}_test.jsonl"
        write_jsonl(out_file, verified)
        logger.info(f"[{code}] Saved {len(verified)} 100% verified sports articles -> {out_file} (Screened: {screened_count})")

        final_summary.append({
            "language": code,
            "total_candidates_pool": len(match_pool),
            "candidates_screened": screened_count,
            "verified_true_sports": len(verified),
            "retention_rate_pct": round(len(verified) / screened_count * 100, 2) if screened_count else 0.0
        })

    summary_df = pd.DataFrame(final_summary)
    summary_csv = resolve_path(cfg, "outputs_dir", "sports_filtering_audit.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info("=== SPORTS DATASET FINAL AUDIT ===")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
