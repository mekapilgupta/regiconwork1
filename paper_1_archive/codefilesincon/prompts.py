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
        "this schema (no markdown fences, no extra keys, no commentary):\n\n"
        "{{\n"
        '  "entities_source": [\n'
        "    {{\n"
        '      "surface_form": string,   // MUST be an exact, verbatim substring of the ORIGINAL question below\n'
        '      "entity_type": string,    // e.g. PERSON, LOCATION, ORG, WORK, DATE, MISC\n'
        '      "romanized_form": string, // Latin-script transliteration of surface_form\n'
        '      "english_form": string,   // English name/translation of the entity, if known FROM THE TEXT ONLY\n'
        '      "confidence": number      // 0.0-1.0\n'
        "    }}\n"
        "  ],\n"
        '  "relation": string,           // the relation/predicate being asked about, in English, short phrase\n'
        '  "answer_type": string | null, // expected answer type, e.g. PERSON, DATE, NUMBER, LOCATION; null if unclear\n'
        '  "constraints": [\n'
        "    {{\n"
        '      "text_source": string,       // exact substring of the original question expressing a constraint\n'
        '      "normalized_english": string,// English normalization of that constraint\n'
        '      "constraint_type": string,   // e.g. TEMPORAL, LOCATION, QUANTITY, QUALIFIER\n'
        '      "confidence": number\n'
        "    }}\n"
        "  ],\n"
        '  "uncertain_items": []          // list any surface forms/constraints you were unsure how to categorize\n'
        "}}\n\n"
        "CRITICAL: every `surface_form` and every `text_source` value MUST be "
        "copied verbatim, character-for-character, from the ORIGINAL question "
        "text below — do not translate, romanize, or normalize them in those "
        "two fields specifically (translation/romanization goes in the other "
        "fields).\n\n"
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


# LLM-judge prompt (separate from the frozen retrieval prompts — spec §7,
# judge labels are explicitly "silver, not ground truth" and can iterate
# independently, though we still version+cache it).
JUDGE_PROMPT_VERSION = "judge_v1"
JUDGE_PROMPT = (
    "You are a strict relevance judge for a retrieval evaluation. You will "
    "be shown a QUESTION (in its original language, with an English gloss), "
    "structured FACETS extracted from it, and a batch of PASSAGES. The "
    "system identity that produced each passage ranking is hidden from you. "
    "For each passage, judge independently — do not compare passages to each "
    "other, do not use outside/world knowledge, do not reason step by step. "
    "Return ONLY a JSON array, one object per passage, in the same order as "
    "given, each with EXACTLY these keys:\n"
    "{{\n"
    '  "passage_id": string,\n'
    '  "passage_supports_answer": boolean,\n'
    '  "relevance": number,        // 0-3\n'
    '  "entity_status": string,    // MATCH | PARTIAL | MISMATCH | NA\n'
    '  "constraint_status": string,// MATCH | PARTIAL | MISMATCH | NA\n'
    '  "translation_issue": boolean,\n'
    '  "confidence": number,       // 0.0-1.0\n'
    '  "short_reason": string      // <= 20 words\n'
    "}}\n\n"
    'Question (original): "{question}"\n'
    'English gloss: "{gloss}"\n'
    "Facets (JSON): {facets_json}\n\n"
    "Passages:\n{passages_block}\n\n"
    "Respond with ONLY the JSON array."
)
