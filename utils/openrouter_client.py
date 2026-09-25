"""Minimal OpenRouter chat-completions client, wired to BudgetTracker.

Every call:
  1. Asks the tracker "can I afford the worst case of this call?" -> refuses if not.
  2. Fires the completion.
  3. Polls OpenRouter's /generation endpoint for the REAL cost of that specific
     generation id (falls back to usage-token estimate if polling fails/disabled).
  4. Records the real cost in the tracker, which raises BudgetExceededError the
     instant cumulative spend would cross the cap.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests

from .budget_tracker import BudgetTracker


class OpenRouterClient:
    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key_env_var: str,
        tracker: BudgetTracker,
        poll_generation_cost: bool = True,
        request_timeout_s: int = 60,
    ) -> None:
        try:
            from dotenv import load_dotenv
            from .io_utils import REPO_ROOT
            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass

        api_key = os.environ.get(api_key_env_var) or os.environ.get("LLM_API_KEY_PAPER2") or os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env_var} (or LLM_API_KEY_PAPER2 / LLM_API_KEY / OPENROUTER_API_KEY) is not set. "
                f"Export your OpenRouter API key before running the judge stage."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.tracker = tracker
        self.poll_generation_cost = poll_generation_cost
        self.timeout_s = request_timeout_s

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter for attribution/rate limiting:
            "X-Title": "indic-sports-summarization-factuality",
        }

    def _poll_real_cost(self, generation_id: str, retries: int = 5, delay_s: float = 1.0) -> float | None:
        """OpenRouter's cost accounting can lag slightly behind the completion
        response; retry briefly before giving up and falling back to estimate."""
        for _ in range(retries):
            try:
                resp = requests.get(
                    f"{self.base_url}/generation",
                    params={"id": generation_id},
                    headers=self._headers(),
                    timeout=self.timeout_s,
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    cost = data.get("total_cost")
                    if cost is not None:
                        return float(cost)
            except requests.RequestException:
                pass
            time.sleep(delay_s)
        return None

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_output_tokens: int,
        temperature: float = 0.0,
        call_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict:
        """Returns {"text": str, "cost_usd": float, "raw": dict}.
        Raises BudgetExceededError (propagated from tracker) if the cap would
        be crossed. Raises RuntimeError on affordability pre-check failure."""
        if not self.tracker.can_afford(max_output_tokens):
            raise RuntimeError(
                f"Refusing to call {self.model_id}: worst-case cost would exceed "
                f"remaining budget (${self.tracker.remaining_usd:.4f} left). "
                f"{self.tracker.summary()}"
            )

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        retries = 3
        last_err = None
        data = {}
        text = ""
        generation_id = None
        for attempt in range(retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    text = msg.get("content") or msg.get("reasoning") or ""
                generation_id = data.get("id")
                break
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (2 ** attempt))
        else:
            if last_err:
                raise last_err

        if not text and "error" in data:
            raise RuntimeError(f"OpenRouter API error: {data['error']}")

        real_cost = None
        if self.poll_generation_cost and generation_id:
            real_cost = self._poll_real_cost(generation_id)

        if real_cost is None:
            # Fallback: estimate from usage tokens using the same conservative
            # ceiling the tracker uses for pre-call checks (pessimistic on purpose).
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", max_output_tokens)
            real_cost = (completion_tokens / 1000.0) * self.tracker.conservative_price_per_1k_output_tokens

        cid = call_id or generation_id or f"call_{time.time()}"
        self.tracker.record_call(cid, self.model_id, real_cost, meta=meta)

        return {"text": text, "cost_usd": real_cost, "raw": data}
