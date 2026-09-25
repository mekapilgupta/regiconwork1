"""Shared helpers: config loading, jsonl read/write, path resolution, logging."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path = "configs/pipeline.yaml") -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str, *parts: str) -> Path:
    """Resolve a path under one of the configured base dirs, creating it if needed.

    Example: resolve_path(cfg, "outputs_dir", "filtered", "hindi_sports.jsonl")
    """
    base = REPO_ROOT / cfg["paths"][key]
    full = base.joinpath(*parts)
    full.parent.mkdir(parents=True, exist_ok=True)
    return full


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_logger(name: str, cfg: dict[str, Any] | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if cfg is not None:
        log_dir = REPO_ROOT / cfg["paths"]["logs_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def set_offline_env() -> None:
    """Force every HF library to refuse network calls and ensure torch.load compatibility.
    Call this at the top of every phase-2 (GPU/offline) script."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Bypass transformers 4.41+ torch.load safety check on torch < 2.6
    try:
        import transformers.modeling_utils
        import transformers.utils.import_utils
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass
