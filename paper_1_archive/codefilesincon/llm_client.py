"""
Single entry point for ALL LLM calls: call_llm(prompt_version, input, ...).

Rules enforced here (spec §2):
- temperature=0
- JSON-schema / structured-output mode when the prompt kind needs JSON
  (facets, judge); OpenRouter forwards `response_format` to models that
  support it (Qwen/Gemini via OpenRouter both do structured JSON output).
- max_output_tokens=800
- Every response is cached to cache/llm/<sha256(model,prompt_version,input)>.json
  and a cached hash is NEVER re-called.
- Retries only on invalid JSON, max 2 retries (3 attempts total).
- Token counts (input/output) are recorded per call in the cache file and
  summed against BUDGET_USD; once 90% of budget is spent, ALL further LLM
  calls raise BudgetExceeded.
- Runs async (httpx.AsyncClient) with a bounded semaphore so you can fan out
  over hundreds of queries without hammering the API or your own machine.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from src.config import SETTINGS, LLM_CACHE_DIR, STATE_DIR, get_logger
from src.utils_io import sha256_key
from src import prompts as P

log = get_logger("llm_client")

COST_STATE_FILE = STATE_DIR / "llm_cost_spent.json"

# Rough per-model $ / 1K tokens fallback pricing table, used only if
# OpenRouter doesn't return usage.cost in the response (it usually does).
# Kept intentionally conservative (slightly high) so BUDGET_USD is respected.
_FALLBACK_PRICE_PER_1K = {"input": 0.002, "output": 0.006}


class InvalidJSONError(Exception):
    pass


class BudgetExceeded(Exception):
    pass


@dataclass
class LLMResult:
    prompt_version: str
    model: str
    input_key: str
    text: str
    parsed_json: Optional[Any]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    latency_s: float


def _cache_path(model: str, prompt_version: str, input_str: str) -> Path:
    key = sha256_key(model, prompt_version, input_str)
    return LLM_CACHE_DIR / f"{key}.json"


def _load_cost_state() -> float:
    if COST_STATE_FILE.exists():
        return json.loads(COST_STATE_FILE.read_text(encoding="utf-8")).get("spent_usd", 0.0)
    return 0.0


def _save_cost_state(spent: float) -> None:
    COST_STATE_FILE.write_text(json.dumps({"spent_usd": spent}, indent=2), encoding="utf-8")


_cost_lock = asyncio.Lock()


async def _check_and_reserve_budget(estimated_cost: float) -> None:
    async with _cost_lock:
        spent = _load_cost_state()
        cap = SETTINGS.budget_usd * 0.9
        if spent >= cap:
            raise BudgetExceeded(
                f"90% of BUDGET_USD (${SETTINGS.budget_usd}) already spent "
                f"(${spent:.4f}). All further LLM calls are blocked."
            )
        if spent + estimated_cost > cap:
            log.warning(
                "This call would push spend past the 90%% budget cap "
                "(spent=$%.4f, cap=$%.4f) — proceeding since it's already committed, "
                "but this will likely be the last call.", spent, cap
            )


async def _record_cost(actual_cost: float) -> None:
    async with _cost_lock:
        spent = _load_cost_state() + actual_cost
        _save_cost_state(spent)


def _needs_json(kind: str) -> bool:
    return kind in ("facets", "judge")


async def _call_openrouter(client: httpx.AsyncClient, prompt: str, kind: str) -> dict:
    headers = {
        "Authorization": f"Bearer {SETTINGS.llm_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",  # OpenRouter recommends setting these
        "X-Title": "clir-pilot",
    }
    body: dict[str, Any] = {
        "model": SETTINGS.llm_model,
        "temperature": 0,
        "max_tokens": 800,
        "messages": [{"role": "user", "content": prompt}],
    }
    if _needs_json(kind):
        body["response_format"] = {"type": "json_object"}

    resp = await client.post(
        f"{SETTINGS.openrouter_base_url}/chat/completions",
        headers=headers,
        json=body,
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_text_and_usage(raw: dict) -> tuple[str, int, int, float]:
    text = raw["choices"][0]["message"]["content"]
    usage = raw.get("usage", {}) or {}
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    cost = usage.get("cost")
    if cost is None:
        cost = (in_tok / 1000) * _FALLBACK_PRICE_PER_1K["input"] + \
               (out_tok / 1000) * _FALLBACK_PRICE_PER_1K["output"]
    return text, in_tok, out_tok, float(cost)


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0]
    return t.strip()


async def call_llm(
    kind: str,
    prompt_version: str,
    input_payload: dict,
    prompt_text: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    require_json: Optional[bool] = None,
) -> LLMResult:
    """
    kind: one of "translate", "rewrite", "facets", "judge" — used for
          cache-key namespacing and to decide JSON mode.
    input_payload: the logical input (dict) hashed into the cache key,
          via a stable json dump — NOT the rendered prompt text, so that
          prompt formatting whitespace changes don't bust the cache
          spuriously... actually per spec the cache key IS
          (model, prompt_version, input) where "input" is the query input,
          so we hash a canonical json of input_payload.
    prompt_text: the fully rendered prompt to send.
    """
    P.assert_not_mutated_after_freeze(kind if kind in ("translate", "rewrite", "facets") else kind, prompt_version)

    input_str = json.dumps(input_payload, sort_keys=True, ensure_ascii=False)
    cache_path = _cache_path(SETTINGS.llm_model, prompt_version, input_str)

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        parsed = cached.get("parsed_json")
        return LLMResult(
            prompt_version=prompt_version, model=SETTINGS.llm_model,
            input_key=cache_path.stem, text=cached["text"], parsed_json=parsed,
            input_tokens=cached["input_tokens"], output_tokens=cached["output_tokens"],
            cost_usd=cached["cost_usd"], cached=True, latency_s=0.0,
        )

    want_json = require_json if require_json is not None else _needs_json(kind)
    max_retries = 2
    last_err: Optional[Exception] = None

    async with semaphore:
        await _check_and_reserve_budget(estimated_cost=0.01)  # conservative pre-check
        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            raw = await _call_openrouter(client, prompt_text, kind)
            latency = time.monotonic() - t0
            text, in_tok, out_tok, cost = _extract_text_and_usage(raw)

            parsed_json = None
            if want_json:
                try:
                    parsed_json = json.loads(_strip_json_fences(text))
                except json.JSONDecodeError as e:
                    last_err = InvalidJSONError(str(e))
                    log.warning("Invalid JSON on attempt %d/%d for %s: %s",
                                attempt + 1, max_retries + 1, kind, e)
                    if attempt < max_retries:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    else:
                        raise last_err

            await _record_cost(cost)
            result = LLMResult(
                prompt_version=prompt_version, model=SETTINGS.llm_model,
                input_key=cache_path.stem, text=text, parsed_json=parsed_json,
                input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost,
                cached=False, latency_s=latency,
            )
            cache_path.write_text(json.dumps({
                "kind": kind, "prompt_version": prompt_version,
                "model": SETTINGS.llm_model, "input": input_payload,
                "text": text, "parsed_json": parsed_json,
                "input_tokens": in_tok, "output_tokens": out_tok,
                "cost_usd": cost, "latency_s": latency,
                "timestamp": time.time(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return result

    raise last_err or RuntimeError("unreachable")


async def list_models(client: httpx.AsyncClient) -> list[str]:
    headers = {"Authorization": f"Bearer {SETTINGS.llm_api_key}"}
    resp = await client.get(f"{SETTINGS.openrouter_base_url}/models", headers=headers, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    return [m["id"] for m in data.get("data", [])]


async def verify_model_available() -> dict:
    """Call the provider's model-list endpoint; abort if LLM_MODEL isn't
    listed. Returns a dict with the matched entry (includes version string
    if the provider reports one)."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {SETTINGS.llm_api_key}"}
        resp = await client.get(f"{SETTINGS.openrouter_base_url}/models", headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        for m in data:
            if m.get("id") == SETTINGS.llm_model:
                log.info("LLM_MODEL '%s' confirmed available. version=%s",
                          SETTINGS.llm_model, m.get("id"))
                return m
    raise RuntimeError(
        f"LLM_MODEL='{SETTINGS.llm_model}' not found in provider model list. "
        f"Aborting per spec §0."
    )
