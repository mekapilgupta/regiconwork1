"""
Prompt templates. Per spec section 2: iterate on 20 queries only, then FREEZE
`prompt_version` and never edit the frozen version's text again.

Each prompt_version below is a literal, versioned string. To iterate, add a
NEW version (e.g. "facets_v2") — never mutate an existing one, or the cache
keys (which hash prompt_version + input) would silently start returning
stale-but-differently-produced results under the same name.

`freeze_prompt_version()` writes the active version ids to state/, and
llm_client.call_llm() refuses to run if a *different* version than the frozen
one is requested after freezing.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import STATE_DIR

FREEZE_FILE = STATE_DIR / "frozen_prompt_versions.json"

ACTIVE_VERSIONS = {
    "translate": "translate_v1",
    "rewrite": "rewrite_v1",
    "facets": "facets_v1",
}

_COMMON_GUARD = (
    "You are a strict text-transformation tool, not a question-answering "
    "assistant. Rules that MUST NOT be broken:\n"
    "1. Do NOT answer the question.\n"
    "2. Do NOT add any world knowledge, facts, or entities that are not "
    "present in the source text.\n"
    "3. Do NOT invent entities, dates, numbers, or names.\n"
    "4. If something is ambiguous, keep it ambiguous rather than resolving it "
    "with outside knowledge.\n"
)

PROMPTS: dict[str, str] = {
    "translate_v1": (
        _COMMON_GUARD
        + "\nTask: Produce a single faithful English translation of the "
        "question below. One sentence. Preserve meaning exactly; do not "
        "paraphrase beyond what translation requires.\n\n"
        'Source language: {lang}\nQuestion: "{question}"\n\n'
        "Respond with ONLY the English translation, no quotes, no preamble."
    ),
    "translate_gold_v1": (
        _COMMON_GUARD
        + "\nTask: You are an expert bilingual human translator producing an authoritative English reference translation. "
        "Translate the following question into precise, natural, canonical English while maintaining all named entities in standard English encyclopedic format.\n\n"
        'Source language: {lang}\nQuestion: "{question}"\n\n'
        "Respond with ONLY the English translation, no quotes, no preamble."
    ),
    "multi_view_v1": (
        _COMMON_GUARD
        + "\nTask: Generate 7 distinct, grammatically complete textual views for cross-lingual retrieval. "
        "CRITICAL RULES:\n"
        "- Every view MUST be a full, complete, grammatical sentence or prose passage.\n"
        "- Do NOT answer the question. Do NOT hallucinate entities not mentioned in the source.\n"
        "- In `cloze_declarative` and `hyde_passage`, use `____` for the requested unknown entity.\n\n"
        "Return ONLY a single JSON object matching this schema:\n"
        "{{\n"
        '  "trans_literal": "Faithful literal English translation sentence",\n'
        '  "trans_natural": "Fluent, natural encyclopedic English translation sentence",\n'
        '  "trans_alias": "English paraphrase using common historical/geographical aliases (e.g. Bombay for Mumbai)",\n'
        '  "cloze_declarative": "Declarative English assertion with ____ in place of the answer",\n'
        '  "src_paraphrase": "Grammatically complete question paraphrase in the SOURCE language",\n'
        '  "src_romanized": "Full sentence Romanized Latin transliteration of the source question",\n'
        '  "hyde_passage": "A 2-3 sentence introductory Wikipedia-style passage about the topic, using ____ for the answer"\n'
        "}}\n\n"
        'Source language: {lang}\nQuestion: "{question}"\n\n'
        "Respond with ONLY the JSON object."
    ),
    "rewrite_v1": (
        _COMMON_GUARD
        + "\nTask: You are given an ENGLISH translation of a question "
        "(already produced separately). Produce a faithful English REWRITE "
        "of it — same meaning, different surface wording — to serve as a "
        "control for LLM rewording effects. Do not change entities, dates, "
        "numbers, or the type of thing being asked for.\n\n"
        'English translation: "{translation}"\n\n'
        "Respond with ONLY the rewritten English sentence, no quotes, no preamble."
    ),
    "facets_v1": (
        _COMMON_GUARD
        + "\nTask: Extract structured facets from the ORIGINAL-LANGUAGE "
        "question below. Return ONLY a single JSON object matching exactly "
        "this schema (no markdown fences, no extra keys, no commentary, no inline comments):\n\n"
        "{{\n"
        '  "entities_source": [\n'
        "    {{\n"
        '      "surface_form": "exact verbatim substring of the original question",\n'
        '      "entity_type": "PERSON | LOCATION | ORG | WORK | DATE | MISC",\n'
        '      "romanized_form": "Latin transliteration",\n'
        '      "english_form": "English translation of entity from text only",\n'
        '      "confidence": 0.95\n'
        "    }}\n"
        "  ],\n"
        '  "relation": "short English phrase describing relation/predicate",\n'
        '  "answer_type": "PERSON | DATE | NUMBER | LOCATION | null",\n'
        '  "constraints": [\n'
        "    {{\n"
        '      "text_source": "exact substring of constraint",\n'
        '      "normalized_english": "English normalization of constraint",\n'
        '      "constraint_type": "TEMPORAL | LOCATION | QUANTITY | QUALIFIER",\n'
        '      "confidence": 0.95\n'
        "    }}\n"
        "  ],\n"
        '  "uncertain_items": []\n'
        "}}\n\n"
        "CRITICAL: every `surface_form` and every `text_source` value MUST be "
        "copied verbatim, character-for-character, from the ORIGINAL question "
        "text below.\n\n"
        'Source language: {lang}\nOriginal question: "{question}"\n\n'
        "Respond with ONLY the JSON object."
    ),
}

FACETS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities_source": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "surface_form": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "romanized_form": {"type": "string"},
                    "english_form": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["surface_form", "entity_type", "english_form", "confidence"],
            },
        },
        "relation": {"type": "string"},
        "answer_type": {"type": ["string", "null"]},
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text_source": {"type": "string"},
                    "normalized_english": {"type": "string"},
                    "constraint_type": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["text_source", "normalized_english", "constraint_type", "confidence"],
            },
        },
        "uncertain_items": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["entities_source", "relation", "answer_type", "constraints", "uncertain_items"],
}


def render(prompt_version: str, **kwargs) -> str:
    if prompt_version not in PROMPTS:
        raise KeyError(f"Unknown prompt_version '{prompt_version}'")
    return PROMPTS[prompt_version].format(**kwargs)


def freeze_prompt_versions() -> dict:
    FREEZE_FILE.write_text(json.dumps(ACTIVE_VERSIONS, indent=2), encoding="utf-8")
    return ACTIVE_VERSIONS


def get_frozen_prompt_versions() -> dict:
    if not FREEZE_FILE.exists():
        return {}
    return json.loads(FREEZE_FILE.read_text(encoding="utf-8"))


def assert_not_mutated_after_freeze(kind: str, requested_version: str) -> None:
    frozen = get_frozen_prompt_versions()
    if not frozen:
        return  # not frozen yet, e.g. still in the 20-query iteration phase
    expected = frozen.get(kind)
    if expected and requested_version != expected:
        raise RuntimeError(
            f"prompt_version mismatch after freeze: kind={kind} "
            f"requested={requested_version} frozen={expected}. "
            f"Prompts are frozen after the 20-query iteration phase (spec §2) "
            f"and must not change mid-pilot."
        )


# LLM-judge prompt (versioned independently from retrieval prompts).
JUDGE_PROMPT_VERSION = "judge_v3"
JUDGE_PROMPT = (
    "You are an objective expert relevance judge for Cross-Lingual Information Retrieval (CLIR) QA evaluation.\n"
    "You will evaluate retrieved English Wikipedia passages for a given question.\n\n"
    "INPUT CONTEXT:\n"
    '- Question (Original): "{question}"\n'
    '- English Gloss: "{gloss}"\n'
    "- Structured Facets: {facets_json}\n\n"
    "EVALUATION CRITERIA:\n"
    "1. `exact_query_copy`: Set to true ONLY if the passage merely echoes or restates the question with no substantive factual information. For normal informative Wikipedia articles, this MUST be false.\n"
    "2. `llm_supports_answer`: Set to true if the passage provides verifiable factual evidence that answers the core question.\n"
    "3. `relevance`: Score from 0 to 3:\n"
    "   - 0: Not relevant / no useful information / completely off-topic.\n"
    "   - 1: Marginally related or general background on the entity/topic, but does not answer the question.\n"
    "   - 2: Relevant evidence / partially answers the question or provides strong contextual proof.\n"
    "   - 3: Direct and definitive answer to the question with clear supporting evidence.\n"
    "4. `entity_status` & `constraint_status`: 'MATCH' if entities/constraints in question are satisfied, else 'MISMATCH' or 'NA'.\n\n"
    "Passages to Evaluate:\n{passages_block}\n\n"
    "Return ONLY a valid JSON array of objects (one per passage in the exact order presented) with keys:\n"
    "[\n"
    "  {{\n"
    '    "passage_id": "string",\n'
    '    "exact_query_copy": boolean,\n'
    '    "llm_supports_answer": boolean,\n'
    '    "relevance": number,\n'
    '    "entity_status": "MATCH" | "MISMATCH" | "NA",\n'
    '    "constraint_status": "MATCH" | "MISMATCH" | "NA",\n'
    '    "confidence": number,\n'
    '    "short_reason": "string (<= 20 words explaining factual match or lack thereof)"\n'
    "  }}\n"
    "]"
)
