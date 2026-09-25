"""Persistent, hard-capped USD budget tracker for the OpenRouter judging stage.

Design goals:
- The cap survives interrupted/resumed runs (state is written to disk after
  every single call, not just at exit).
- We check BEFORE every call whether we can afford the worst-case cost of
  that call (estimated from max_output_tokens * a conservative price ceiling),
  and refuse to fire it if not.
- We reconcile with REAL cost after every call (via OpenRouter's generation
  endpoint, see openrouter_client.py) so the running total is accurate, not
  just estimated.
- Going over the cap is a hard error, not a warning: the pipeline stops.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class BudgetTracker:
    state_path: Path
    max_total_usd: float
    safety_margin_usd: float = 0.25
    # Conservative per-call worst-case price ceiling used only for the
    # pre-call affordability check, in USD per 1K output tokens. Overridden
    # per-model if you know exact pricing; default is deliberately pessimistic.
    conservative_price_per_1k_output_tokens: float = 0.02

    _spent_usd: float = field(default=0.0, init=False)
    _call_log: list[dict] = field(default_factory=list, init=False)
    _lock: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self.state_path = Path(self.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._spent_usd = state.get("spent_usd", 0.0)
            self._call_log = state.get("call_log", [])

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_total_usd - self._spent_usd)

    def _persist(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {"spent_usd": self._spent_usd, "call_log": self._call_log},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def can_afford(self, max_output_tokens: int) -> bool:
        with self._lock:
            worst_case = (max_output_tokens / 1000.0) * self.conservative_price_per_1k_output_tokens
            return (self._spent_usd + worst_case + self.safety_margin_usd) <= self.max_total_usd

    def record_call(self, call_id: str, model_id: str, real_cost_usd: float, meta: dict | None = None) -> None:
        with self._lock:
            self._spent_usd += real_cost_usd
            self._call_log.append(
                {
                    "call_id": call_id,
                    "model_id": model_id,
                    "cost_usd": real_cost_usd,
                    "cumulative_usd": self._spent_usd,
                    "timestamp": time.time(),
                    "meta": meta or {},
                }
            )
            self._persist()
            if self._spent_usd > self.max_total_usd:
                raise BudgetExceededError(
                    f"Budget exceeded after call {call_id}: spent ${self._spent_usd:.4f} "
                    f"> cap ${self.max_total_usd:.2f}. Stopping immediately."
                )

    def summary(self) -> str:
        with self._lock:
            return (
                f"Spent ${self._spent_usd:.4f} of ${self.max_total_usd:.2f} cap "
                f"({len(self._call_log)} calls, ${self.remaining_usd:.4f} remaining)."
            )
