"""
Offline dense embedding generation.

- Loads pinned model (BAAI/bge-m3 or intfloat/multilingual-e5-small) from local HF cache.
- Batched GPU encoding with mixed precision (fp16/bf16) if available, CPU fallback.
- Caching of query and passage embeddings to cache/emb/.
- Absolutely NO network access (assert_offline).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.config import EMB_CACHE_DIR, HF_CACHE_DIR, assert_offline, get_logger, get_pinned_embedding_model
from src.utils_io import sha256_key

log = get_logger("embeddings")


class DenseEncoder:
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        assert_offline()
        pin = get_pinned_embedding_model()
        self.model_id = model_id or pin["model_id"]

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        log.info("Loading encoder '%s' on device %s (offline)...", self.model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=True,
        )
        self.model.eval()
        self.model.to(self.device)
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            self.model.to(torch.bfloat16)
        elif self.device.startswith("cuda"):
            self.model.to(torch.float16)

    def _prepare_text(self, text: str, is_query: bool = True) -> str:
        text = text.strip()
        if "e5" in self.model_id.lower():
            prefix = "query: " if is_query else "passage: "
            if not text.startswith(prefix):
                return prefix + text
        return text

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        is_query: bool = True,
        normalize: bool = True,
        max_length: int = 512,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        prepared = [self._prepare_text(t, is_query=is_query) for t in texts]
        all_embeddings = []

        for i in range(0, len(prepared), batch_size):
            batch_texts = prepared[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)

            if "bge" in self.model_id.lower() and hasattr(outputs, "last_hidden_state"):
                # BGE dense representation uses [CLS] (first token)
                sentence_embeddings = outputs.last_hidden_state[:, 0]
            else:
                # E5 and standard encoders use mean pooling
                sentence_embeddings = self._mean_pooling(outputs, encoded["attention_mask"])

            if normalize:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            all_embeddings.append(sentence_embeddings.cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)

    def encode_cached(
        self,
        text: str,
        is_query: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        h = sha256_key(self.model_id, "query" if is_query else "passage", text)
        cache_file = EMB_CACHE_DIR / f"{h}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        emb = self.encode([text], batch_size=1, is_query=is_query, normalize=normalize)[0]
        np.save(cache_file, emb)
        return emb
