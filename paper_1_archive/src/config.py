"""
Central configuration.

- Reads env vars (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, BUDGET_USD) via python-dotenv.
- Never logs or prints the API key.
- Exposes assert_offline() / assert_online() guards so GPU-stage scripts
  cannot accidentally make network calls, and CPU-stage scripts fail loudly
  if run somewhere without network.
- Detects GPU once, at stage-1 time, and persists the decision to
  state/embedding_model.json so every later stage uses the SAME model
  (never re-decided mid-pilot, which would break comparability).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
LLM_CACHE_DIR = CACHE_DIR / "llm"
EMB_CACHE_DIR = CACHE_DIR / "emb"
HF_CACHE_DIR = CACHE_DIR / "hf"
PYSERINI_CACHE_DIR = CACHE_DIR / "pyserini"
RUNS_DIR = ROOT / "runs"
METRICS_DIR = ROOT / "metrics"
ANALYSIS_DIR = ROOT / "analysis"
PLOTS_DIR = ROOT / "plots"
STATE_DIR = ROOT / "state"
CONFIG_SNAPSHOT_DIR = ROOT / "config"

for d in (DATA_DIR, CACHE_DIR, LLM_CACHE_DIR, EMB_CACHE_DIR, HF_CACHE_DIR,
          PYSERINI_CACHE_DIR, RUNS_DIR, METRICS_DIR, ANALYSIS_DIR, PLOTS_DIR,
          STATE_DIR, CONFIG_SNAPSHOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")


def _mask(s: Optional[str]) -> str:
    if not s:
        return "<unset>"
    return s[:4] + "…" + s[-2:] if len(s) > 8 else "***"


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "openrouter"))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    budget_usd: float = field(default_factory=lambda: float(os.environ.get("BUDGET_USD", "5")))
    openrouter_base_url: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )

    def validate(self):
        if not self.llm_model:
            raise RuntimeError("LLM_MODEL is not set in .env")
        if not self.llm_api_key or self.llm_api_key == "REPLACE_ME":
            raise RuntimeError("LLM_API_KEY is not set in .env")
        if self.budget_usd <= 0:
            raise RuntimeError("BUDGET_USD must be > 0")

    def safe_repr(self) -> dict:
        """Never include the raw key."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_api_key_masked": _mask(self.llm_api_key),
            "budget_usd": self.budget_usd,
            "openrouter_base_url": self.openrouter_base_url,
        }


SETTINGS = Settings()


# ---------------------------------------------------------------------------
# Offline / online guards
# ---------------------------------------------------------------------------

def _network_reachable(host: str = "openrouter.ai", port: int = 443, timeout: float = 2.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


def assert_offline():
    """Call at the top of any GPU-stage script. Refuses to run if the
    process appears to have real internet (i.e. you launched it on the
    wrong node)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if os.environ.get("CLIR_FORCE_ONLINE_OK") == "1":
        return  # explicit escape hatch for local dev/testing
    if _network_reachable():
        raise RuntimeError(
            "network access attempted/detected in an offline (GPU) stage. "
            "This script must run on the offline GPU node, downstream of "
            "scripts/02_download_cpu_online.py and 03_llm_stage_cpu_online.py. "
            "If this is a false positive (e.g. local dev), set "
            "CLIR_FORCE_ONLINE_OK=1."
        )


def assert_online():
    """Call at the top of any CPU-stage script that needs the LLM API or
    downloads. Fails fast with a clear message rather than hanging."""
    if not _network_reachable():
        raise RuntimeError(
            "no network detected. This script must run on the online CPU "
            "node. If you're on the GPU node by mistake, switch nodes."
        )


# ---------------------------------------------------------------------------
# Embedding model pin (decided once at stage 1, reused everywhere)
# ---------------------------------------------------------------------------

EMBED_MODEL_PIN_FILE = STATE_DIR / "embedding_model.json"
GPU_MODEL = "BAAI/bge-m3"
CPU_MODEL = "intfloat/multilingual-e5-small"


def pin_embedding_model(has_gpu: bool) -> dict:
    model_id = GPU_MODEL if has_gpu else CPU_MODEL
    payload = {"model_id": model_id, "has_gpu": has_gpu, "pinned_by": "01_check_env.py"}
    EMBED_MODEL_PIN_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def get_pinned_embedding_model() -> dict:
    if not EMBED_MODEL_PIN_FILE.exists():
        raise RuntimeError(
            "No embedding model pinned yet. Run scripts/01_check_env.py first "
            "(it detects GPU presence once and pins bge-m3 or e5-small for the "
            "whole pilot — the model must never change mid-run)."
        )
    return json.loads(EMBED_MODEL_PIN_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class _RedactKeyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        key = SETTINGS.llm_api_key
        if key and key in str(record.msg):
            record.msg = str(record.msg).replace(key, _mask(key))
        return True


for _h in logging.getLogger().handlers:
    _h.addFilter(_RedactKeyFilter())
