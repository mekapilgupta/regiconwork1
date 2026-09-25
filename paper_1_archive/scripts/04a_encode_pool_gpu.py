#!/usr/bin/env python3
"""
Pure GPU Dense Pool Encoder (No PyJNIus, No JVM - 100% PyTorch CUDA).

Encodes candidate pool texts (~77k for bn, ~66k for te) with BAAI/bge-m3 on GPU.
Saves:
  cache/pool/bn/pool_emb.npy
  cache/pool/te/pool_emb.npy
"""
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.utils_io import read_jsonl

POOL_CACHE_DIR = ROOT / "cache" / "pool"


def get_local_model_path(model_id: str = "BAAI/bge-m3") -> str:
    cache_hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if cache_hub.exists():
        for d in cache_hub.iterdir():
            if d.is_dir() and (d / "modules.json").exists():
                return str(d)
    repo_hub = ROOT / "cache" / "hf" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if repo_hub.exists():
        for d in repo_hub.iterdir():
            if d.is_dir() and (d / "modules.json").exists():
                return str(d)
    return model_id


def encode_track(track: str, model: SentenceTransformer, batch_size: int = 128):
    track_pool_dir = POOL_CACHE_DIR / track
    pool_text_file = track_pool_dir / "pool_text.jsonl"
    out_emb_file = track_pool_dir / "pool_emb.npy"

    assert pool_text_file.exists(), f"Pool text file not found: {pool_text_file}"
    passages = list(read_jsonl(pool_text_file))
    texts = [p["text"] for p in passages]

    print(f"\n[*] Encoding {len(texts):,} passages for track '{track}' (batch_size={batch_size})...")
    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed = time.time() - start_time
    print(f"[✓] Encoded {len(texts):,} passages in {elapsed:.1f}s ({len(texts)/elapsed:.1f} pass/sec).")

    np.save(out_emb_file, embeddings.astype(np.float32))
    print(f"[✓] Saved {out_emb_file} (shape={embeddings.shape})")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing model on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model_path = get_local_model_path("BAAI/bge-m3")
    print(f"[*] Loading model from: {model_path}")
    model = SentenceTransformer(model_path, device=device)
    model.max_seq_length = 512
    if device == "cuda":
        model = model.half()

    for track in ["bn", "te"]:
        encode_track(track, model, batch_size=128)

    print("\n[✓] ALL POOLS ENCODED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
