#!/usr/bin/env python
"""Prepares a balanced, diverse multilingual news evaluation dataset across
English, Hindi, and Bengali from the official XL-Sum test splits.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_utils import get_logger, load_config, resolve_path, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of test articles per language")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("prepare_eval_data", cfg)
    seed = cfg["xlsum"]["random_seed"]
    rng = random.Random(seed)

    out_dir = resolve_path(cfg, "data_dir", "sports_filtered")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for lang in cfg["languages"]:
        code = lang["code"]
        raw_dir = resolve_path(cfg, "data_dir", "xlsum_raw", code)
        if not raw_dir.exists():
            logger.warning(f"Raw dir not found: {raw_dir}")
            continue

        ds = load_from_disk(str(raw_dir))
        test_split = ds.get("test", [])
        articles = [dict(row) for row in test_split]

        if len(articles) > args.sample_size:
            rng.shuffle(articles)
            selected = articles[: args.sample_size]
        else:
            selected = articles

        out_file = out_dir / f"{code}_test.jsonl"
        write_jsonl(out_file, selected)
        logger.info(f"[{code}] Extracted {len(selected)} articles from test split ({len(articles)} total) -> {out_file}")

        summary.append({
            "language": code,
            "total_test_split_size": len(articles),
            "sampled_eval_size": len(selected)
        })

    print("\n" + str(summary))


if __name__ == "__main__":
    main()
