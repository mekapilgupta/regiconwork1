"""Text processing utilities for multilingual evaluation.
Includes special-token stripping, Unicode NFC normalization, script detection,
and generation validity classification.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter


def strip_special_tokens(text: str) -> str:
    """Removes special/formatting tokens from model outputs while preserving meaningful text."""
    if not text:
        return ""
    # Strip HTML-like / language / special tokens e.g. <s>, </s>, <2bn>, <pad>, <unk>
    cleaned = re.sub(r"<(?:/?[a-zA-Z0-9_\-]+)>", "", text)
    # Strip bracketed tokens e.g. [CLS], [SEP], [PAD], [MASK]
    cleaned = re.sub(r"\[(?:CLS|SEP|PAD|UNK|MASK)\]", "", cleaned)
    # Normalize unicode
    cleaned = unicodedata.normalize("NFC", cleaned)
    return cleaned.strip()


def classify_generation_status(raw_text: str, target_lang: str) -> str:
    """Classifies the generation status into one of:
    - 'empty'
    - 'malformed'
    - 'wrong_script'
    - 'truncated'
    - 'generation_error'
    - 'valid'
    """
    if not raw_text or not raw_text.strip():
        return "empty"
    
    cleaned = strip_special_tokens(raw_text)
    if not cleaned:
        return "malformed"
        
    # Check for truncated output (< 10 chars or < 3 words)
    if len(cleaned) < 10 or len(cleaned.split()) < 3:
        return "truncated"
        
    # Check for degenerate punctuation repetition e.g. '-------', '.......'
    if re.search(r"([^\w\s])\1{5,}", raw_text):
        return "malformed"
        
    # Check for degenerate word/token repetition (e.g., 'ख़ ख़ ख़')
    words = cleaned.split()
    if len(words) >= 6:
        word_counts = Counter(words)
        _, top_count = word_counts.most_common(1)[0]
        if top_count / len(words) > 0.5:
            return "malformed"
            
    # Check for unmapped placeholder / CJK corruption artifacts from tokenizer mismatch
    cjk_artifacts = len(re.findall(r"[\u4e00-\u9fff\ufffd\u3000-\u303f]", raw_text))
    if cjk_artifacts >= 2:
        return "malformed"
        
    # Check script mismatch
    if target_lang == "bengali":
        bengali_chars = len(re.findall(r"[\u0980-\u09FF]", cleaned))
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", cleaned))
        latin_chars = len(re.findall(r"[a-zA-Z]", cleaned))
        if bengali_chars == 0 and (devanagari_chars > 5 or latin_chars > 10):
            return "wrong_script"
    elif target_lang == "hindi":
        bengali_chars = len(re.findall(r"[\u0980-\u09FF]", cleaned))
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", cleaned))
        if devanagari_chars == 0 and bengali_chars > 5:
            return "wrong_script"
            
    return "valid"
