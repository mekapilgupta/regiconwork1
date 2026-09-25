#!/usr/bin/env python
"""RUN ON THE GPU NODE (offline). Extracts a small, fixed inventory of
sports-relevant fact types from BOTH the source article and the generated
summary, for every (system, language) pair:

  - athlete_name  : PERSON entities (spaCy for English, IndicNER for Indic)
  - team_name     : ORG entities
  - score_number  : regex over digit patterns (scores, distances, times, etc.)
  - date          : regex over common date patterns
  - winner_keyword_context : heuristic — the entity token(s) nearest to a
    winner/loser cue word ("won", "beat", "defeated", "lost to", "hara",
    "जीता", "हराया", etc.), used only as an approximate winner signal. This
    is explicitly a heuristic, not a relation-extraction model, and must be
    described as such in the paper's limitations section.

Kept deliberately simple (no training, no relation-extraction model) per the
1-2 hour feasibility constraint. Writes one facts.jsonl per (system, language)
with source_facts and summary_facts per example, ready for 06_fact_consistency.py.
"""
from __future__ import annotations

import argparse
import glob
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

# --- regex fact types -------------------------------------------------------

_NUMBER_RE = re.compile(
    r"""
    \b\d+[.,]?\d*          # plain numbers / decimals: 89.94, 2, 1000
    (?:\s?-\s?\d+[.,]?\d*)? # optional score pattern: 2-1, 3 - 0
    \b
    """,
    re.VERBOSE,
)

_DATE_RE = re.compile(
    r"""
    \b(?:\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s?\d{0,4})\b
    |
    \b(?:\d{4}-\d{2}-\d{2})\b
    |
    \b(?:\d{1,2}/\d{1,2}/\d{2,4})\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Winner/loser cue words: English + a small hand-built Hindi/Bengali list.
# Extend this list per language as needed; it is intentionally minimal.
_WINNER_CUES = {
    "english": ["won", "beat", "defeated", "victory", "triumphed", "clinched"],
    "hindi": ["जीता", "हराया", "जीत", "पराजित"],
    "bengali": ["জিতেছে", "হারিয়েছে", "জয়"],
}
_LOSER_CUES = {
    "english": ["lost to", "defeated by", "lost", "beaten by"],
    "hindi": ["हार", "हारा", "पराजय"],
    "bengali": ["হেরেছে", "পরাজিত"],
}


def extract_numbers(text: str) -> list[str]:
    return [m.group(0).strip() for m in _NUMBER_RE.finditer(text)]


def extract_dates(text: str) -> list[str]:
    return [m.group(0).strip() for m in _DATE_RE.finditer(text)]


def extract_winner_context_entities(text: str, entities: list[tuple[str, str]], lang_code: str) -> list[str]:
    """Very simple proximity heuristic: for each winner cue word occurrence,
    take the nearest PERSON/ORG entity by character distance. This is a
    heuristic proxy for a winner relation, not a trained relation extractor —
    documented explicitly as a limitation."""
    cues = _WINNER_CUES.get(lang_code, [])
    hits = []
    for cue in cues:
        for m in re.finditer(re.escape(cue), text):
            cue_pos = m.start()
            nearest = None
            nearest_dist = None
            for ent_text, ent_label in entities:
                if ent_label not in ("PERSON", "ORG"):
                    continue
                pos = text.find(ent_text)
                if pos == -1:
                    continue
                dist = abs(pos - cue_pos)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest = ent_text
            if nearest:
                hits.append(nearest)
    return hits


# --- NER backends ------------------------------------------------------------


class SpacyNER:
    def __init__(self, model_name: str):
        import spacy

        self.nlp = spacy.load(model_name)

    def extract(self, text: str) -> list[tuple[str, str]]:
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ("PERSON", "ORG")]


class IndicNerHF:
    """Wraps ai4bharat/IndicNER via a local transformers token-classification
    pipeline, offline. Maps model-specific tag scheme to PERSON/ORG."""

    def __init__(self, local_dir: Path):
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(str(local_dir), local_files_only=True)
        self.pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
        # NER label scheme uses PER/ORG/LOC; map to our two labels of interest.
        self._label_map = {"PER": "PERSON", "ORG": "ORG", "PERSON": "PERSON"}

    def extract(self, text: str) -> list[tuple[str, str]]:
        # Guard against pathologically long inputs for the token classifier.
        truncated = text[:2000]
        results = self.pipe(truncated)
        out = []
        for r in results:
            mapped = self._label_map.get(r.get("entity_group", ""), None)
            if mapped:
                out.append((r["word"], mapped))
        return out


def build_ner_backend(cfg: dict, lang: dict):
    backend = lang["ner_backend"]
    if backend == "spacy":
        return SpacyNER(cfg["ner_models"]["spacy"]["spacy_model_name"])
    elif backend == "indicner":
        local_dir = REPO_ROOT / cfg["ner_models"]["indicner"]["local_dir"]
        return IndicNerHF(local_dir)
    else:
        raise ValueError(f"Unknown ner_backend: {backend}")


def extract_facts_for_text(text: str, ner_backend, lang_code: str) -> dict:
    entities = ner_backend.extract(text)
    persons = sorted({e for e, label in entities if label == "PERSON"})
    orgs = sorted({e for e, label in entities if label == "ORG"})
    numbers = extract_numbers(text)
    dates = extract_dates(text)
    winner_ctx = extract_winner_context_entities(text, entities, lang_code)
    return {
        "person_name": persons,
        "org_name": orgs,
        "numeric_quantity": numbers,
        "athlete_name": persons,
        "team_name": orgs,
        "score_number": numbers,
        "date": dates,
        "winner_keyword_context": sorted(set(winner_ctx)),
    }


def main() -> None:
    set_offline_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("fact_extraction", cfg)

    summaries_dir = REPO_ROOT / cfg["paths"]["outputs_dir"] / "summaries"
    files = sorted(glob.glob(str(summaries_dir / "*.jsonl")))
    if not files:
        logger.warning(f"No summary files found in {summaries_dir}. Run 03_run_summarization.py first.")
        return

    lang_lookup = {l["code"]: l for l in cfg["languages"]}
    ner_cache: dict[str, object] = {}

    for fpath in files:
        fname = Path(fpath).stem
        system, lang_code = fname.split("__")
        lang = lang_lookup[lang_code]

        if lang["ner_backend"] not in ner_cache:
            logger.info(f"Loading NER backend '{lang['ner_backend']}' for {lang_code}")
            ner_cache[lang["ner_backend"]] = build_ner_backend(cfg, lang)
        ner_backend = ner_cache[lang["ner_backend"]]

        rows = list(read_jsonl(fpath))
        logger.info(f"[{system}/{lang_code}] extracting facts for {len(rows)} examples")

        enriched = []
        for row in rows:
            source_facts = extract_facts_for_text(row["source_text"], ner_backend, lang_code)
            summary_facts = extract_facts_for_text(row["generated_summary"], ner_backend, lang_code)
            reference_facts = extract_facts_for_text(row["reference_summary"], ner_backend, lang_code)
            enriched.append(
                {
                    "id": row.get("id"),
                    "system": system,
                    "language": lang_code,
                    "source_facts": source_facts,
                    "summary_facts": summary_facts,
                    "reference_facts": reference_facts,
                }
            )

        out_path = resolve_path(cfg, "outputs_dir", "facts", f"{fname}__facts.jsonl")
        write_jsonl(out_path, enriched)
        logger.info(f"[{system}/{lang_code}] wrote fact tuples -> {out_path}")


if __name__ == "__main__":
    main()
