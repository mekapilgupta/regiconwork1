"""
Single entry point for ALL LLM calls: call_llm(prompt_version, input, ...).

Rules enforced here:
- temperature=0
- JSON-schema / structured-output mode when the prompt kind needs JSON
  (facets, judge).
- Handles thinking/reasoning models (qwen3.7-flash, etc.) by allocating
  max_tokens=2048, stripping <think> tags, and extracting final answers.
- Provider fallback enabled on OpenRouter to navigate upstream shared pool rate limits.
- Resilient exponential backoff on 429 rate limits, timeouts, and connection errors.
- Every response is cached to cache/llm/<sha256(model,prompt_version,input)>.json
  and a cached hash is NEVER re-called.
- Retries on invalid JSON (up to 3 attempts total).
- Token counts (input/output) recorded per call in the cache file and
  summed against BUDGET_USD.
- Runs async (httpx.AsyncClient) with a bounded semaphore.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from src.config import SETTINGS, LLM_CACHE_DIR, STATE_DIR, get_logger
from src.utils_io import sha256_key
from src import prompts as P

log = get_logger("llm_client")

COST_STATE_FILE = STATE_DIR / "llm_cost_spent.json"

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


async def _record_cost(actual_cost: float) -> None:
    async with _cost_lock:
        spent = _load_cost_state() + actual_cost
        _save_cost_state(spent)


def _needs_json(kind: str) -> bool:
    return kind in ("facets",)


def _clean_thinking_tags(text: str) -> str:
    if not text:
        return ""
    # Strip <think>...</think> tags if produced by reasoning/thinking models.
    # Some models (qwen3.7-flash) produce thinking tags even in json_object mode.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also strip /think without closing tag (truncated)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


async def _call_openrouter(client: httpx.AsyncClient, prompt: str, kind: str) -> dict:
    headers = {
        "Authorization": f"Bearer {SETTINGS.llm_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "clir-pilot",
    }
    # For thinking models (qwen3.7-flash etc.), do NOT use json_object response_format
    # as it causes the model to output thinking tags inside the JSON content field,
    # breaking JSON parsing. We instead do post-hoc extraction.
    is_thinking_model = any(
        tok in SETTINGS.llm_model.lower()
        for tok in ("qwen3", "qwq", "deepseek-r", "o1", "o3")
    )
    body: dict[str, Any] = {
        "model": SETTINGS.llm_model,
        "temperature": 0,
        "max_tokens": 4096,  # thinking models need more for think + answer
        "messages": [{"role": "user", "content": prompt}],
        "provider": {
            "allow_fallbacks": True
        }
    }
    if _needs_json(kind) and not is_thinking_model:
        body["response_format"] = {"type": "json_object"}

    max_network_retries = 8
    backoffs = [5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0]

    for attempt in range(max_network_retries + 1):
        try:
            resp = await client.post(
                f"{SETTINGS.openrouter_base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=120.0,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                sleep_s = backoffs[min(attempt, len(backoffs) - 1)]
                log.warning(
                    "OpenRouter returned status %d on attempt %d/%d. Retrying in %.1fs...",
                    resp.status_code,
                    attempt + 1,
                    max_network_retries,
                    sleep_s,
                )
                if attempt < max_network_retries:
                    await asyncio.sleep(sleep_s)
                    continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
            sleep_s = backoffs[min(attempt, len(backoffs) - 1)]
            log.warning(
                "Network error calling OpenRouter (%s) on attempt %d/%d. Retrying in %.1fs...",
                e,
                attempt + 1,
                max_network_retries,
                sleep_s,
            )
            if attempt < max_network_retries:
                await asyncio.sleep(sleep_s)
                continue
            raise e

    raise RuntimeError("Max retries exceeded connecting to OpenRouter.")


def _extract_text_and_usage(raw: dict) -> tuple[str, str, int, int, float]:
    """Returns (cleaned_text, raw_content, in_tok, out_tok, cost)."""
    msg = raw.get("choices", [{}])[0].get("message", {})
    raw_content = msg.get("content") or msg.get("reasoning") or ""
    text = _clean_thinking_tags(raw_content)
    # Also check for reasoning field (some OpenRouter providers put thinking there)
    if not text and msg.get("reasoning"):
        raw_content = msg["reasoning"]
        text = _clean_thinking_tags(raw_content)

    usage = raw.get("usage", {}) or {}
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    cost = usage.get("cost")
    if cost is None:
        cost = (in_tok / 1000) * _FALLBACK_PRICE_PER_1K["input"] + (out_tok / 1000) * _FALLBACK_PRICE_PER_1K["output"]
    return text, raw_content, in_tok, out_tok, float(cost)


def _clean_json_str(s: str) -> str:
    # Strip single-line comments // ...
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    # Strip multi-line comments /* ... */
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    # Strip trailing commas before closing braces/brackets
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s.strip()


def _extract_json_object(text: str) -> Optional[Any]:
    if not text:
        return None
    t = text.strip()

    # 1. Try markdown code block
    if "```" in t:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
        if match:
            candidate = _clean_json_str(match.group(1))
            try:
                return json.loads(candidate)
            except Exception:
                pass

    # 2. Try direct load
    try:
        return json.loads(_clean_json_str(t))
    except Exception:
        pass

    # 3. Check for array [ ... ] vs object { ... } based on which starts first
    first_sq = t.find("[")
    last_sq = t.rfind("]")
    first_b = t.find("{")
    last_b = t.rfind("}")

    # If array appears first or only array exists
    if first_sq != -1 and last_sq != -1 and last_sq > first_sq:
        if first_b == -1 or first_sq < first_b:
            try:
                return json.loads(_clean_json_str(t[first_sq : last_sq + 1]))
            except Exception:
                pass

    # If object exists
    if first_b != -1 and last_b != -1 and last_b > first_b:
        try:
            return json.loads(_clean_json_str(t[first_b : last_b + 1]))
        except Exception:
            pass

    # Fallback to array if object failed
    if first_sq != -1 and last_sq != -1 and last_sq > first_sq:
        try:
            return json.loads(_clean_json_str(t[first_sq : last_sq + 1]))
        except Exception:
            pass

    return None


async def call_llm(
    kind: str,
    prompt_version: str,
    input_payload: dict,
    prompt_text: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    require_json: Optional[bool] = None,
) -> LLMResult:
    P.assert_not_mutated_after_freeze(kind if kind in ("translate", "rewrite", "facets") else kind, prompt_version)

    input_str = json.dumps(input_payload, sort_keys=True, ensure_ascii=False)
    cache_path = _cache_path(SETTINGS.llm_model, prompt_version, input_str)

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        parsed = cached.get("parsed_json")
        return LLMResult(
            prompt_version=prompt_version,
            model=SETTINGS.llm_model,
            input_key=cache_path.stem,
            text=cached["text"],
            parsed_json=parsed,
            input_tokens=cached.get("input_tokens", 0),
            output_tokens=cached.get("output_tokens", 0),
            cost_usd=cached.get("cost_usd", 0.0),
            cached=True,
            latency_s=0.0,
        )

    want_json = require_json if require_json is not None else _needs_json(kind)
    max_retries = 2
    last_err: Optional[Exception] = None

    async with semaphore:
        await _check_and_reserve_budget(estimated_cost=0.01)
        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            raw = await _call_openrouter(client, prompt_text, kind)
            latency = time.monotonic() - t0
            text, raw_content, in_tok, out_tok, cost = _extract_text_and_usage(raw)

            parsed_json = None
            if want_json:
                parsed_json = _extract_json_object(text)
                if parsed_json is None and raw_content and raw_content != text:
                    # For thinking models: JSON appears after </think> in raw_content
                    parsed_json = _extract_json_object(raw_content)
                if parsed_json is None:
                    last_err = InvalidJSONError(f"Failed to parse JSON | Text: {text[:200]}")
                    log.warning(
                        "Invalid JSON on attempt %d/%d for %s: Text starts with: %s",
                        attempt + 1,
                        max_retries + 1,
                        kind,
                        text[:150],
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(3.0 * (attempt + 1))
                        continue
                    else:
                        # Don't raise — return empty dict so pipeline continues
                        log.error(
                            "All JSON retries exhausted for %s. Saving empty facets and continuing.",
                            kind,
                        )
                        parsed_json = {}

            await _record_cost(cost)
            result = LLMResult(
                prompt_version=prompt_version,
                model=SETTINGS.llm_model,
                input_key=cache_path.stem,
                text=text,
                parsed_json=parsed_json,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                cached=False,
                latency_s=latency,
            )
            cache_path.write_text(
                json.dumps({
                    "kind": kind,
                    "prompt_version": prompt_version,
                    "model": SETTINGS.llm_model,
                    "input": input_payload,
                    "text": text,
                    "parsed_json": parsed_json,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost,
                    "latency_s": latency,
                    "timestamp": time.time(),
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return result

    raise last_err or RuntimeError("unreachable")


async def verify_model_available() -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {SETTINGS.llm_api_key}"}
        resp = await client.get(f"{SETTINGS.openrouter_base_url}/models", headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        for m in data:
            if m.get("id") == SETTINGS.llm_model:
                log.info("LLM_MODEL '%s' confirmed available.", SETTINGS.llm_model)
                return m
    raise RuntimeError(
        f"LLM_MODEL='{SETTINGS.llm_model}' not found in OpenRouter model list."
    )
