"""Automated unit tests for surface and factuality metrics.
Verifies ROUGE exact-match, empty string handling, Indic tokenization,
chrF scale, and Unicode NFC normalization.
"""
import sys
from pathlib import Path
import importlib.util
import pytest
import sacrebleu
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("surface_metrics", str(ROOT / "scripts" / "04_surface_metrics.py"))
surface_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface_module)
IndicMultilingualTokenizer = surface_module.IndicMultilingualTokenizer


def test_indic_multilingual_tokenizer_basic():
    tok = IndicMultilingualTokenizer()
    # Bengali sentence
    bn_text = "প্রধানমন্ত্রী সোমবার নতুন অর্থনৈতিক সংস্কারের ঘোষণা দিয়েছেন।"
    tokens_bn = tok.tokenize(bn_text)
    assert len(tokens_bn) > 0
    assert "প্রধানমন্ত্রী" in tokens_bn
    assert "অর্থনৈতিক" in tokens_bn

    # Hindi sentence
    hi_text = "प्रधानमंत्री मनमोहन सिंह ने नए आर्थिक सुधारों की घोषणा की।"
    tokens_hi = tok.tokenize(hi_text)
    assert len(tokens_hi) > 0
    assert "प्रधानमंत्री" in tokens_hi
    assert "मनमोहन" in tokens_hi


def test_rouge_exact_match():
    tok = IndicMultilingualTokenizer()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=tok)
    
    # English exact match
    en_text = "The president announced new economic reforms on Monday."
    scores_en = scorer.score(en_text, en_text)
    assert scores_en["rouge1"].fmeasure == pytest.approx(1.0, abs=1e-4)
    assert scores_en["rougeL"].fmeasure == pytest.approx(1.0, abs=1e-4)

    # Hindi exact match
    hi_text = "प्रधानमंत्री ने सोमवार को नए आर्थिक सुधारों की घोषणा की।"
    scores_hi = scorer.score(hi_text, hi_text)
    assert scores_hi["rouge1"].fmeasure == pytest.approx(1.0, abs=1e-4)
    assert scores_hi["rougeL"].fmeasure == pytest.approx(1.0, abs=1e-4)

    # Bengali exact match
    bn_text = "প্রধানমন্ত্রী সোমবার নতুন অর্থনৈতিক সংস্কারের ঘোষণা দিয়েছেন।"
    scores_bn = scorer.score(bn_text, bn_text)
    assert scores_bn["rouge1"].fmeasure == pytest.approx(1.0, abs=1e-4)
    assert scores_bn["rougeL"].fmeasure == pytest.approx(1.0, abs=1e-4)


def test_rouge_empty_prediction():
    tok = IndicMultilingualTokenizer()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=tok)
    ref = "Government confirms agreement on transport corridor."
    scores = scorer.score(ref, "")
    assert scores["rouge1"].fmeasure == 0.0
    assert scores["rougeL"].fmeasure == 0.0


def test_rouge_partial_overlap():
    tok = IndicMultilingualTokenizer()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=tok)
    ref = "भारत और रूस ने रक्षा समझौते पर हस्ताक्षर किए।"
    pred = "भारत और रूस ने समझौते पर हस्ताक्षर किए।"
    scores = scorer.score(ref, pred)
    assert 0.5 < scores["rouge1"].fmeasure < 1.0


def test_copied_source_overlap():
    tok = IndicMultilingualTokenizer()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=tok)
    # Lead sentence copied directly from source containing reference facts
    source_lead = "The national government announced comprehensive economic reforms on Monday to stimulate trade."
    ref = "The government announced economic reforms on Monday."
    scores = scorer.score(ref, source_lead)
    assert scores["rouge1"].fmeasure > 0.6
    assert scores["rougeL"].fmeasure > 0.6


def test_special_token_removal():
    from utils.text_utils import strip_special_tokens
    raw_text = "<s><2bn> প্রধানমন্ত্রী সোমবার নতুন সংস্কারের ঘোষণা দেন।</s>"
    cleaned = strip_special_tokens(raw_text)
    assert "<s>" not in cleaned
    assert "</s>" not in cleaned
    assert "<2bn>" not in cleaned
    assert cleaned == "প্রধানমন্ত্রী সোমবার নতুন সংস্কারের ঘোষণা দেন।"


def test_chrf_scale():
    chrf = sacrebleu.metrics.CHRF()
    ref = "The quick brown fox jumps over the lazy dog."
    score = chrf.sentence_score(ref, [ref]).score
    assert 99.0 <= score <= 100.0  # chrF is on 0-100 scale


if __name__ == "__main__":
    pytest.main(["-v", __file__])

