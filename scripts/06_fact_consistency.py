#!/usr/bin/env python
"""RUN ON THE GPU NODE (offline). Computes fact-tuple consistency between
generated summaries and their SOURCE ARTICLES (not the reference summary).

This is a deliberate design choice, not an oversight: XL-Sum references are
known to contain "extra information" not inferable from the source article
(Hasan et al. 2021), so scoring a generated summary's facts against the
reference would conflate two different problems (model hallucination vs.
reference hallucination). Restricting the gold fact set to what is
verifiably present in the source article is exactly the control the
novelty-risk analysis flagged as necessary.

For each fact type we report, per example:
  - summary_fact_precision: fraction of facts in the SUMMARY that also
    appear in the SOURCE (i.e., not fabricated)
  - summary_fact_recall: fraction of facts in the SOURCE that also appear
    in the SUMMARY (i.e., not dropped)
  - exact_match string comparison, case-normalized, since sports facts
    (names, numbers) need exact identity, not fuzzy overlap

We additionally report a single derived flag per example:
  - winner_flip_suspected: True if the summary's winner_keyword_context
    entities share NO overlap with the source's, despite the summary
    containing at least one winner-cue-adjacent entity. This is the
    approximate signal for "the summary named a different winner" and is
    the headline error type for the paper's analysis (Category C framing:
    "ROUGE stayed high while the winner changed").
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
    write_jsonl,
)


def _norm(items: list[str]) -> set[str]:
    return {i.strip().lower() for i in items if i.strip()}


def precision_recall(summary_items: list[str], source_items: list[str]) -> dict:
    s = _norm(summary_items)
    src = _norm(source_items)
    if not s and not src:
        return {"precision": None, "recall": None, "n_summary": 0, "n_source": 0}
    overlap = s & src
    precision = (len(overlap) / len(s)) if s else None
    recall = (len(overlap) / len(src)) if src else None
    return {
        "precision": precision,
        "recall": recall,
        "n_summary": len(s),
        "n_source": len(src),
        "n_overlap": len(overlap),
    }


def winner_flip_suspected(summary_ctx: list[str], source_ctx: list[str]) -> bool | None:
    s = _norm(summary_ctx)
    src = _norm(source_ctx)
    if not s or not src:
        return None  # not enough signal either side; do not count as a flip
    return len(s & src) == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("fact_consistency", cfg)

    facts_dir = REPO_ROOT / cfg["paths"]["outputs_dir"] / "facts"
    files = sorted(glob.glob(str(facts_dir / "*.jsonl")))
    if not files:
        logger.warning(f"No fact files found in {facts_dir}. Run 05_fact_extraction.py first.")
        return

    fact_types = [ft for ft in cfg["fact_types"] if ft != "winner_keyword_context"]
    all_rows_for_csv = []

    for fpath in files:
        fname = Path(fpath).stem  # e.g. "mt5_xlsum__hindi__facts"
        system_lang = fname.replace("__facts", "")
        system, lang_code = system_lang.split("__")

        rows = list(read_jsonl(fpath))
        logger.info(f"[{system}/{lang_code}] scoring fact consistency for {len(rows)} examples")

        enriched = []
        for row in rows:
            per_type = {}
            for ft in fact_types:
                # Safe lookup with fallback for alias names
                s_facts = row["summary_facts"].get(ft)
                if s_facts is None:
                    fallback_map = {
                        "person_name": "athlete_name",
                        "org_name": "team_name",
                        "numeric_quantity": "score_number",
                        "athlete_name": "person_name",
                        "team_name": "org_name",
                        "score_number": "numeric_quantity"
                    }
                    alt_key = fallback_map.get(ft, ft)
                    s_facts = row["summary_facts"].get(alt_key, [])
                    src_facts = row["source_facts"].get(alt_key, [])
                else:
                    src_facts = row["source_facts"].get(ft, [])
                pr = precision_recall(s_facts, src_facts)
                per_type[ft] = pr

            flip = winner_flip_suspected(
                row["summary_facts"].get("winner_keyword_context", []),
                row["source_facts"].get("winner_keyword_context", []),
            )

            enriched.append(
                {
                    "id": row["id"],
                    "system": system,
                    "language": lang_code,
                    "fact_consistency": per_type,
                    "winner_flip_suspected": flip,
                }
            )

            csv_row = {"system": system, "language": lang_code, "id": row["id"], "winner_flip_suspected": flip}
            for ft in fact_types:
                csv_row[f"{ft}_precision"] = per_type[ft]["precision"]
                csv_row[f"{ft}_recall"] = per_type[ft]["recall"]
            all_rows_for_csv.append(csv_row)

        out_path = resolve_path(cfg, "outputs_dir", "fact_consistency", f"{system_lang}__consistency.jsonl")
        write_jsonl(out_path, enriched)
        logger.info(f"[{system}/{lang_code}] wrote fact consistency -> {out_path}")

    import pandas as pd

    df = pd.DataFrame(all_rows_for_csv)
    summary_csv = resolve_path(cfg, "outputs_dir", "fact_consistency_summary.csv")
    metric_cols = [c for c in df.columns if c.endswith("_precision") or c.endswith("_recall")]
    agg = df.groupby(["system", "language"])[metric_cols].mean()

    flip_rate = df.groupby(["system", "language"])["winner_flip_suspected"].apply(
        lambda s: s.dropna().mean() if s.notna().any() else None
    )
    agg["winner_flip_suspected_rate"] = flip_rate

    agg.to_csv(summary_csv)
    logger.info(f"Wrote per-system/per-language fact consistency summary -> {summary_csv}")
    logger.info(
        "NOTE: winner_flip_suspected is a heuristic proximity signal, not a "
        "validated relation extractor. Treat it as a triage signal for "
        "selecting examples for human/LLM-judge review in phase 3, not as a "
        "standalone factuality metric in the paper's headline results."
    )


if __name__ == "__main__":
    main()
