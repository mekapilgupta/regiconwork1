#!/usr/bin/env python3
"""
Stage 1: Environment and Sanity Checks (CPU, Online).

- Checks network access to OpenRouter.
- Validates API key and model availability.
- Detects GPU presence and pins embedding model (bge-m3 or e5-small) to state/embedding_model.json.
- Freezes initial prompt versions.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import (
    SETTINGS,
    assert_online,
    get_logger,
    pin_embedding_model,
)
from src.llm_client import verify_model_available
from src.prompts import freeze_prompt_versions
from src.utils_io import mark_done

log = get_logger("01_check_env")


async def main_async():
    log.info("=== Stage 1: Environment Sanity Check ===")
    assert_online()
    log.info("Network connectivity to OpenRouter confirmed.")

    # Check Settings
    log.info("Checking configuration: %s", SETTINGS.safe_repr())
    if not SETTINGS.llm_api_key or SETTINGS.llm_api_key.startswith("sk-or-REPLACE"):
        log.warning("LLM_API_KEY is not set or using placeholder in .env. Please update .env before running LLM stages.")
    else:
        try:
            matched_model = await verify_model_available()
            log.info("LLM model verified: %s", matched_model.get("id"))
        except Exception as e:
            log.error("LLM model check failed: %s", e)
            log.warning("Continuing environment check; please verify LLM_MODEL in .env.")

    # Check GPU / PyTorch
    has_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_gpu else "CPU only (Login/CPU Node)"
    log.info("PyTorch %s | CUDA Available: %s | Device: %s", torch.__version__, has_gpu, device_name)

    # Pin embedding model
    pin = pin_embedding_model(has_gpu=has_gpu)
    log.info("Pinned embedding model for this pilot: %s", pin["model_id"])

    # Freeze prompt versions
    frozen = freeze_prompt_versions()
    log.info("Prompt versions frozen: %s", frozen)

    # Mark stage done
    for track in ["bn", "te", "hi", "hi_mono"]:
        mark_done(ROOT / "state", "stage01", track, {"pinned_model": pin["model_id"], "has_gpu": has_gpu})

    log.info("Stage 1 completed successfully.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
