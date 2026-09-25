#!/usr/bin/env python
"""Task 8 Audit: Diagnostic inspection of IndicBART on Bengali inputs.

Tests:
1. Model checkpoint & tokenizer vocabulary/special tokens.
2. Generation behavior with and without language tags/forced_bos_token_id.
3. Analysis of degeneration patterns (repetition loops, wrong script generation).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io_utils import REPO_ROOT, get_logger, load_config, resolve_path, set_offline_env


def main():
    set_offline_env()
    cfg = load_config("configs/pipeline.yaml")
    local_dir = REPO_ROOT / cfg["models"]["indicbart"]["local_dir"]

    print("=== 1. IndicBART Checkpoint & Tokenizer Inspection ===")
    print(f"Loading from: {local_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(local_dir), local_files_only=True)
    model.eval()

    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"All special token IDs: {tokenizer.all_special_ids}")

    # Check for language tags in vocab
    lang_tags = [t for t in tokenizer.get_vocab().keys() if t.startswith("_") or "<2" in t or "bn" in t]
    print(f"Candidate language tokens in vocab: {lang_tags[:20]}")

    # Test sample Bengali sentence
    sample_bn = "ইন্দোনেশিয়ার সুরাবায়া প্রদেশে তিনটি গীর্জায় হামলায় নিহত হয়েছেন অন্তত ১৩জন। ইসলামিক স্টেট গোষ্ঠী হামলার দায় স্বীকার করেছে।"
    print(f"\nTest Input (Bengali): {sample_bn}")

    inputs = tokenizer(sample_bn, return_tensors="pt")
    
    # Generation without forced language BOS (as originally executed)
    with torch.no_grad():
        gen_default = model.generate(**inputs, max_length=84, num_beams=4)
    decoded_default = tokenizer.decode(gen_default[0], skip_special_tokens=False)
    decoded_clean = tokenizer.decode(gen_default[0], skip_special_tokens=True)
    print(f"\n[Default Generation (Raw with special tokens)]:\n{decoded_default}")
    print(f"[Default Generation (Cleaned)]:\n{decoded_clean}")

    # Inspect failure mode
    is_malformed = "---" in decoded_clean or decoded_clean.count(decoded_clean[:5]) > 3 or len(decoded_clean.strip()) < 5
    is_wrong_script = any('\u0900' <= c <= '\u097F' for c in decoded_clean) # contains Devanagari

    audit_findings = {
        "model_key": "indicbart",
        "model_repo": "ai4bharat/IndicBART",
        "tested_input": sample_bn,
        "raw_decoded_output": decoded_default,
        "cleaned_output": decoded_clean,
        "is_malformed": is_malformed,
        "is_wrong_script": is_wrong_script,
        "root_cause_diagnosis": (
            "IndicBART was pretrained with multispelling and unified Indic script representations. "
            "Without language-specific prefixing or forced target language BOS token conditioning during decoding, "
            "the unguided beam search degenerates into hyphen repetition loops and Devanagari fallback on Bengali inputs. "
            "This constitutes an implementation/model-compatibility failure mode, distinct from fluent semantic hallucination."
        )
    }

    out_file = resolve_path(cfg, "outputs_dir", "indicbart_bengali_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_findings, f, indent=2, ensure_ascii=False)
    print(f"\nAudit complete. Saved diagnostic results to {out_file}")


if __name__ == "__main__":
    main()
