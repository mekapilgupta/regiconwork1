#!/usr/bin/env python
"""RUN ON THE GPU NODE (offline). Computes ROUGE-1/2/L, BERTScore, and chrF
for every (system, language) summary file produced by 03_run_summarization.py.

Writes one metrics.jsonl per (system, language) with per-example scores, plus
a combined outputs/surface_metrics_summary.csv with per-system/per-language
means for quick inspection.
"""
from __future__ import annotations

import argparse
import glob
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


import re
import unicodedata
from rouge_score import rouge_scorer, tokenizers


class IndicMultilingualTokenizer(tokenizers.Tokenizer):
    """Script-aware multilingual tokenizer for ROUGE evaluation across Latin and Indic scripts."""
    def tokenize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFC", text or "").strip()
        # Separate punctuation marks with spaces, preserving full Indic word characters and viramas
        text = re.sub(r"([.,!?\"':;(){}\[\]।॥—–\-])", r" \1 ", text)
        tokens = [t.lower() for t in text.split() if t.strip()]
        return tokens


def compute_rouge(rows: list[dict], rouge_types: list[str]) -> list[dict]:
    scorer = rouge_scorer.RougeScorer(rouge_types, tokenizer=IndicMultilingualTokenizer())
    out = []
    for row in rows:
        ref = row.get("reference_summary", "")
        gen = row.get("generated_summary", "")
        scores = scorer.score(ref, gen)
        entry = {t: scores[t].fmeasure for t in rouge_types}
        out.append(entry)
    return out


def compute_chrf(rows: list[dict]) -> list[float]:
    import sacrebleu

    chrf = sacrebleu.metrics.CHRF()
    return [
        chrf.sentence_score(row["generated_summary"], [row["reference_summary"]]).score
        for row in rows
    ]


def compute_bertscore(cfg: dict, lang_code: str, rows: list[dict], logger) -> list[float]:
    from bert_score import BERTScorer

    bcfg = cfg["metrics"]["bertscore"]
    repo_id = bcfg["lang_model_map"].get(lang_code, bcfg["lang_model_map"]["english"])
    local_dir = REPO_ROOT / bcfg["local_dir"] / repo_id.replace("/", "__")

    logger.info(f"[{lang_code}] loading BERTScore backbone from {local_dir}")
    try:
        from transformers import AutoConfig
        m_config = AutoConfig.from_pretrained(str(local_dir), local_files_only=True)
        num_layers = getattr(m_config, "num_hidden_layers", 12)
    except Exception:
        num_layers = 12

    scorer = BERTScorer(model_type=str(local_dir), num_layers=num_layers, lang=None, rescale_with_baseline=False)
    if hasattr(scorer, "_tokenizer") and getattr(scorer._tokenizer, "model_max_length", 0) > 512:
        scorer._tokenizer.model_max_length = 512

    cands = [row["generated_summary"] for row in rows]
    refs = [row["reference_summary"] for row in rows]
    _, _, f1 = scorer.score(cands, refs, batch_size=32)
    return f1.tolist()


def main() -> None:
    set_offline_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("surface_metrics", cfg)

    summaries_dir = REPO_ROOT / cfg["paths"]["outputs_dir"] / "summaries"
    files = sorted(glob.glob(str(summaries_dir / "*.jsonl")))
    if not files:
        logger.warning(f"No summary files found in {summaries_dir}. Run 03_run_summarization.py first.")
        return

    rouge_types = cfg["metrics"]["rouge_types"]
    all_rows_for_csv = []

    for fpath in files:
        fname = Path(fpath).stem  # e.g. "mt5_xlsum__hindi"
        system, lang_code = fname.split("__")
        rows = list(read_jsonl(fpath))
        logger.info(f"[{system}/{lang_code}] scoring {len(rows)} examples")

        rouge_scores = compute_rouge(rows, rouge_types)
        chrf_scores = compute_chrf(rows)
        bert_scores = compute_bertscore(cfg, lang_code, rows, logger)

        enriched = []
        for row, rs, cf, bs in zip(rows, rouge_scores, chrf_scores, bert_scores):
            merged = {**row, **rs, "chrf": cf, "bertscore_f1": bs}
            enriched.append(merged)
            all_rows_for_csv.append(
                {
                    "system": system,
                    "language": lang_code,
                    "id": row.get("id"),
                    **rs,
                    "chrf": cf,
                    "bertscore_f1": bs,
                }
            )

        out_path = resolve_path(cfg, "outputs_dir", "metrics", f"{fname}__surface_metrics.jsonl")
        write_jsonl(out_path, enriched)
        logger.info(f"[{system}/{lang_code}] wrote surface metrics -> {out_path}")

    import pandas as pd

    df = pd.DataFrame(all_rows_for_csv)
    summary_csv = resolve_path(cfg, "outputs_dir", "surface_metrics_summary.csv")
    agg = df.groupby(["system", "language"])[rouge_types + ["chrf", "bertscore_f1"]].mean()
    agg.to_csv(summary_csv)
    logger.info(f"Wrote per-system/per-language mean surface metrics -> {summary_csv}")


if __name__ == "__main__":
    main()
