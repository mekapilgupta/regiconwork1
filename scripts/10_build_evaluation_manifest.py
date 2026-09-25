#!/usr/bin/env python
"""Assembles the single unified, canonical evaluation manifest.
One row per (article_id, language, system) combining all source data,
generation status, surface metrics, fact extraction, and LLM-judge annotations.

Guarantees:
- Every row contains raw and cleaned generations.
- Preserves failed/malformed outputs with explicit status.
- Consistent schema matching reviewer specifications.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_utils import get_logger, load_config, read_jsonl, resolve_path, write_jsonl
from utils.text_utils import classify_generation_status, strip_special_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("build_manifest", cfg)

    # 1. Load verified source articles
    source_articles = {}
    for lang in cfg["languages"]:
        code = lang["code"]
        path = resolve_path(cfg, "data_dir", "sports_filtered", f"{code}_test.jsonl")
        if not path.exists():
            path = resolve_path(cfg, "data_dir", "sports_verified", f"{code}_test.jsonl")
        if path.exists():
            for row in read_jsonl(path):
                source_articles[(code, row["id"])] = row

    # 2. Load generated summaries
    summaries = {}
    for f in glob.glob(str(resolve_path(cfg, "outputs_dir", "summaries") / "*.jsonl")):
        for row in read_jsonl(f):
            summaries[(row["system"], row["language"], row["id"])] = row

    # 3. Load surface metrics
    surface_metrics = {}
    for f in glob.glob(str(resolve_path(cfg, "outputs_dir", "metrics") / "*.jsonl")):
        for row in read_jsonl(f):
            surface_metrics[(row["system"], row["language"], row["id"])] = row

    # 4. Load fact consistency
    fact_consistency = {}
    for f in glob.glob(str(resolve_path(cfg, "outputs_dir", "fact_consistency") / "*.jsonl")):
        for row in read_jsonl(f):
            fact_consistency[(row["system"], row["language"], row["id"])] = row

    # 5. Load LLM judge results
    judge_results = {}
    judge_file = resolve_path(cfg, "outputs_dir", "llm_judge_results.jsonl")
    if judge_file.exists():
        for row in read_jsonl(judge_file):
            judge_results[(row["system"], row["language"], row["id"])] = row

    logger.info(
        f"Loaded: {len(source_articles)} source articles, {len(summaries)} summaries, "
        f"{len(surface_metrics)} metrics, {len(judge_results)} judge records."
    )

    # Assemble canonical manifest rows
    manifest_rows = []
    for (system, lang, aid), sum_row in summaries.items():
        src_row = source_articles.get((lang, aid), {})
        met_row = surface_metrics.get((system, lang, aid), {})
        fc_row = fact_consistency.get((system, lang, aid), {})
        jg_row = judge_results.get((system, lang, aid), {})

        raw_pred = sum_row.get("generated_summary", "")
        cleaned_pred = strip_special_tokens(raw_pred)
        status = classify_generation_status(raw_pred, lang)

        src_text = src_row.get("text", sum_row.get("source_text", ""))
        ref_text = src_row.get("summary", sum_row.get("reference_summary", ""))

        # Factuality Judge fields
        if (system, lang, aid) in judge_results:
            if jg_row.get("verdict") is not None:
                factuality_label = int(jg_row["verdict"])
                judge_status = "judged"
            else:
                factuality_label = None
                judge_status = "judge_parse_error"
        else:
            factuality_label = None
            judge_status = "unjudged"

        flagged_type = jg_row.get("flagged_fact_type")
        # Normalize error type
        if flagged_type == "org_identity":
            flagged_type = "organization_identity"
            
        if status in ["malformed", "wrong_script", "truncated", "empty"]:
            error_level = "output_level"
            error_types = status if flagged_type in [None, "none"] else f"{status},{flagged_type}"
        elif flagged_type and flagged_type != "none":
            error_level = "summary_level"
            error_types = flagged_type
        else:
            error_level = "none"
            error_types = "none"

        row = {
            "article_id": aid,
            "language": lang,
            "system": system,
            "split_source": src_row.get("split_source", "test"),
            "title": src_row.get("title", ""),
            "url": src_row.get("url", ""),
            "generation_status": status,
            "source_text": src_text,
            "reference_summary": ref_text,
            "generated_summary": raw_pred,
            "cleaned_summary": cleaned_pred,
            "source_length_chars": len(src_text),
            "summary_length_chars": len(cleaned_pred),
            "source_length_words": len(src_text.split()),
            "summary_length_words": len(cleaned_pred.split()),
            # Surface overlap metrics
            "rouge1": met_row.get("rouge1"),
            "rouge2": met_row.get("rouge2"),
            "rougeL": met_row.get("rougeL"),
            "chrf": met_row.get("chrf"),
            "bertscore_f1": met_row.get("bertscore_f1"),
            # Fact consistency metrics (Entity / numerical)
            "person_name_precision": fc_row.get("person_name_precision", fc_row.get("athlete_name_precision")),
            "person_name_recall": fc_row.get("person_name_recall", fc_row.get("athlete_name_recall")),
            "org_name_precision": fc_row.get("org_name_precision", fc_row.get("team_name_precision")),
            "org_name_recall": fc_row.get("org_name_recall", fc_row.get("team_name_recall")),
            "numeric_quantity_precision": fc_row.get("numeric_quantity_precision", fc_row.get("score_number_precision")),
            "numeric_quantity_recall": fc_row.get("numeric_quantity_recall", fc_row.get("score_number_recall")),
            "date_precision": fc_row.get("date_precision"),
            "date_recall": fc_row.get("date_recall"),
            # Factuality Judge metrics (LLM Judge)
            "factuality_label": factuality_label,
            "factuality_judge_status": judge_status,
            "error_level": error_level,
            "error_types": error_types,
            "judge_reason": jg_row.get("reason"),
            "judge_cost_usd": jg_row.get("cost_usd"),
        }
        manifest_rows.append(row)

    manifest_jsonl = resolve_path(cfg, "outputs_dir", "evaluation_manifest.jsonl")
    manifest_csv = resolve_path(cfg, "outputs_dir", "evaluation_manifest.csv")

    write_jsonl(manifest_jsonl, manifest_rows)
    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest_csv, index=False)

    logger.info(f"Successfully generated canonical evaluation manifest with {len(manifest_rows)} rows:")
    logger.info(f" -> JSONL: {manifest_jsonl}")
    logger.info(f" -> CSV:   {manifest_csv}")


if __name__ == "__main__":
    main()
