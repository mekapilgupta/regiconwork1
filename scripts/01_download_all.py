#!/usr/bin/env python
"""RUN ON THE CPU LOGIN NODE (must have internet access).

Downloads everything the offline GPU node will ever need:
  - XL-Sum dataset splits for the configured languages
  - Summarization model weights (mT5-XLSum, IndicBART)
  - NER model weights (IndicNER; spaCy English model via `spacy download`)
  - BERTScore backbone weights

Everything is saved as plain files under data/ and models/ using
snapshot_download / dataset.save_to_disk, NOT the usual `from_pretrained`
auto-cache, so phase-2 scripts can load with local_files_only=True and never
attempt a network call.

Idempotent: re-running skips anything already present unless --force is passed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io_utils import REPO_ROOT, get_logger, load_config, resolve_path  # noqa: E402


def download_xlsum(cfg: dict, force: bool) -> None:
    from datasets import load_dataset

    logger = get_logger("download_xlsum", cfg)
    dataset_id = cfg["xlsum"]["hf_dataset_id"]

    for lang in cfg["languages"]:
        out_dir = resolve_path(cfg, "data_dir", "xlsum_raw", lang["code"])
        if out_dir.exists() and any(out_dir.iterdir()) and not force:
            logger.info(f"[skip] {lang['code']} already downloaded at {out_dir}")
            continue
        logger.info(f"Downloading XL-Sum split for {lang['code']} ({lang['xlsum_name']})...")
        ds = load_dataset(dataset_id, lang["xlsum_name"], trust_remote_code=True)
        ds.save_to_disk(str(out_dir))
        logger.info(f"Saved {lang['code']} -> {out_dir} (splits: {list(ds.keys())})")


def snapshot_model(repo_id: str, local_dir: Path, logger, force: bool) -> None:
    from huggingface_hub import snapshot_download

    if local_dir.exists() and any(local_dir.iterdir()) and not force:
        logger.info(f"[skip] {repo_id} already at {local_dir}")
        return
    logger.info(f"Downloading model {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)


def download_summarization_models(cfg: dict, force: bool) -> None:
    logger = get_logger("download_models", cfg)
    for name, mcfg in cfg["models"].items():
        if mcfg.get("type") != "seq2seq":
            continue
        local_dir = REPO_ROOT / mcfg["local_dir"]
        snapshot_model(mcfg["hf_repo_id"], local_dir, logger, force)


def download_ner_models(cfg: dict, force: bool) -> None:
    logger = get_logger("download_ner", cfg)

    # IndicNER via HF snapshot
    indicner_cfg = cfg["ner_models"]["indicner"]
    if indicner_cfg.get("hf_repo_id"):
        local_dir = REPO_ROOT / indicner_cfg["local_dir"]
        snapshot_model(indicner_cfg["hf_repo_id"], local_dir, logger, force)

    # spaCy English model via `python -m spacy download`
    spacy_model = cfg["ner_models"]["spacy"]["spacy_model_name"]
    try:
        import spacy

        spacy.load(spacy_model)
        logger.info(f"[skip] spaCy model {spacy_model} already installed")
    except OSError:
        logger.info(f"Installing spaCy model {spacy_model} ...")
        subprocess.run([sys.executable, "-m", "spacy", "download", spacy_model], check=True)


def download_bertscore_backbone(cfg: dict, force: bool) -> None:
    logger = get_logger("download_bertscore", cfg)
    bcfg = cfg["metrics"]["bertscore"]
    seen = set()
    for lang_key, repo_id in bcfg["lang_model_map"].items():
        if repo_id in seen:
            continue
        seen.add(repo_id)
        # store each distinct backbone under models/bertscore_backbone/<sanitized repo id>
        local_dir = REPO_ROOT / bcfg["local_dir"] / repo_id.replace("/", "__")
        snapshot_model(repo_id, local_dir, logger, force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["dataset", "models", "ner", "bertscore"],
        help="Skip specific download groups",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("download_all", cfg)

    logger.info("=== Phase 1: downloading everything needed for offline GPU run ===")

    if "dataset" not in args.skip:
        download_xlsum(cfg, args.force)
    if "models" not in args.skip:
        download_summarization_models(cfg, args.force)
    if "ner" not in args.skip:
        download_ner_models(cfg, args.force)
    if "bertscore" not in args.skip:
        download_bertscore_backbone(cfg, args.force)

    logger.info(
        "Done. Verify with `du -sh data models`, then copy/rsync this whole "
        "directory to the GPU node's filesystem (or confirm it's already on a "
        "shared mount) before running run_phase2_offline.sh."
    )


if __name__ == "__main__":
    main()
