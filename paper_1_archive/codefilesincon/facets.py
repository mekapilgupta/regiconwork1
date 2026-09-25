"""
Facet post-validation. Per spec §2: `surface_form` must be an exact
substring of the original question; if not, mark facet_invalid (we do this
per-item so a single bad entity doesn't discard an otherwise-good facet set).
"""
from __future__ import annotations

from typing import Any

from src.utils_io import nfc


def _is_exact_substring(candidate: str, source: str) -> bool:
    if not candidate:
        return False
    return nfc(candidate) in nfc(source)


def validate_facets(question: str, facets: dict[str, Any]) -> dict[str, Any]:
    """Returns a copy of `facets` with `facet_invalid: bool` added to each
    entity and constraint item, plus a top-level `any_invalid` summary flag.
    Does not drop invalid items — downstream code decides whether to use
    them (default: they're excluded from S4/S5/S6 construction, see
    src/systems.py)."""
    out = dict(facets)
    any_invalid = False

    entities = []
    for ent in facets.get("entities_source", []) or []:
        ent = dict(ent)
        valid = _is_exact_substring(ent.get("surface_form", ""), question)
        ent["facet_invalid"] = not valid
        any_invalid = any_invalid or not valid
        entities.append(ent)
    out["entities_source"] = entities

    constraints = []
    for c in facets.get("constraints", []) or []:
        c = dict(c)
        valid = _is_exact_substring(c.get("text_source", ""), question)
        c["facet_invalid"] = not valid
        any_invalid = any_invalid or not valid
        constraints.append(c)
    out["constraints"] = constraints

    out["any_invalid"] = any_invalid
    return out


def collect_english_forms(facets: dict[str, Any], valid_only: bool = True) -> list[str]:
    forms = []
    for ent in facets.get("entities_source", []) or []:
        if valid_only and ent.get("facet_invalid"):
            continue
        ef = (ent.get("english_form") or "").strip()
        if ef:
            forms.append(ef)
    return forms


def collect_surface_forms(facets: dict[str, Any], valid_only: bool = True) -> list[str]:
    forms = []
    for ent in facets.get("entities_source", []) or []:
        if valid_only and ent.get("facet_invalid"):
            continue
        sf = (ent.get("surface_form") or "").strip()
        if sf:
            forms.append(sf)
    return forms


def collect_normalized_constraints(facets: dict[str, Any], valid_only: bool = True) -> list[str]:
    out = []
    for c in facets.get("constraints", []) or []:
        if valid_only and c.get("facet_invalid"):
            continue
        ne = (c.get("normalized_english") or "").strip()
        if ne:
            out.append(ne)
    return out


def facets_concat_string(facets: dict[str, Any], valid_only: bool = True) -> str:
    parts = collect_english_forms(facets, valid_only)
    parts.append(facets.get("relation", "") or "")
    parts += collect_normalized_constraints(facets, valid_only)
    return " ".join(p for p in parts if p)
