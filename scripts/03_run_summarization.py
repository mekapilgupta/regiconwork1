#!/usr/bin/env python
"""RUN ON THE GPU NODE (offline). Generates summaries for every configured
system x language, reading only local weights (local_files_only=True /
HF_HUB_OFFLINE=1). Writes one jsonl per (system, language) with the source
article, reference summary, and generated summary.

Systems:
  - lead_baseline: first N sentences of the article (no model needed)
  - mt5_xlsum: csebuetnlp/mT5_multilingual_XLSum, all languages
  - indicbart: ai4bharat/IndicBART, Indic languages only (per config)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io_utils import (  # noqa: E402
    REPO_ROOT,
    get_logger,
    load_config,
    read_jsonl,
    resolve_path,
    set_offline_env,
    write_jsonl,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u0964\u0965])\s+")  # incl. Devanagari danda


def lead_n_summary(text: str, n: int) -> str:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return " ".join(sentences[:n])


def run_lead_baseline(cfg: dict, lang: dict, articles: list[dict], logger) -> list[dict]:
    n = cfg["models"]["lead_baseline"]["n_sentences"]
    out = []
    for row in articles:
        out.append(
            {
                "id": row.get("id"),
                "url": row.get("url"),
                "source_text": row["text"],
                "reference_summary": row["summary"],
                "generated_summary": lead_n_summary(row["text"], n),
                "system": "lead_baseline",
                "language": lang["code"],
            }
        )
    return out


def run_seq2seq_model(cfg: dict, model_key: str, lang: dict, articles: list[dict], logger) -> list[dict]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    mcfg = cfg["models"][model_key]
    local_dir = REPO_ROOT / mcfg["local_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {model_key} from {local_dir} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(local_dir), local_files_only=True).to(device)
    model.eval()

    out = []
    batch_size = 8
    for start in range(0, len(articles), batch_size):
        batch = articles[start : start + batch_size]
        inputs = tokenizer(
            [row["text"] for row in batch],
            max_length=mcfg["max_input_tokens"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=mcfg["max_output_tokens"],
                num_beams=mcfg["num_beams"],
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for row, summary in zip(batch, decoded):
            out.append(
                {
                    "id": row.get("id"),
                    "url": row.get("url"),
                    "source_text": row["text"],
                    "reference_summary": row["summary"],
                    "generated_summary": summary,
                    "system": model_key,
                    "language": lang["code"],
                }
            )
        logger.info(f"[{model_key}/{lang['code']}] generated {start + len(batch)}/{len(articles)}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> None:
    set_offline_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=["lead_baseline", "mt5_xlsum", "indicbart"],
        help="Which systems to run",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("run_summarization", cfg)

    for lang in cfg["languages"]:
        input_path = resolve_path(cfg, "data_dir", "sports_filtered", f"{lang['code']}_test.jsonl")
        if not input_path.exists():
            logger.warning(f"[{lang['code']}] no file at {input_path}, skipping.")
            continue
        articles = list(read_jsonl(input_path))
        logger.info(f"[{lang['code']}] loaded {len(articles)} articles from {input_path.name}.")

        if "lead_baseline" in args.systems:
            results = run_lead_baseline(cfg, lang, articles, logger)
            out_path = resolve_path(cfg, "outputs_dir", "summaries", f"lead_baseline__{lang['code']}.jsonl")
            write_jsonl(out_path, results)
            logger.info(f"[{lang['code']}] lead_baseline -> {out_path}")

        if "mt5_xlsum" in args.systems:
            results = run_seq2seq_model(cfg, "mt5_xlsum", lang, articles, logger)
            out_path = resolve_path(cfg, "outputs_dir", "summaries", f"mt5_xlsum__{lang['code']}.jsonl")
            write_jsonl(out_path, results)
            logger.info(f"[{lang['code']}] mt5_xlsum -> {out_path}")

        if "indicbart" in args.systems:
            applicable = cfg["models"]["indicbart"].get("applicable_languages", [])
            if lang["code"] not in applicable:
                logger.info(f"[{lang['code']}] skipping indicbart (not in applicable_languages)")
            else:
                results = run_seq2seq_model(cfg, "indicbart", lang, articles, logger)
                out_path = resolve_path(cfg, "outputs_dir", "summaries", f"indicbart__{lang['code']}.jsonl")
                write_jsonl(out_path, results)
                logger.info(f"[{lang['code']}] indicbart -> {out_path}")

    logger.info("Done generating summaries for all configured systems/languages.")


if __name__ == "__main__":
    main()
