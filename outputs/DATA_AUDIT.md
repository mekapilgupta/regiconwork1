# DATA AUDIT: Multilingual News Summarization Factuality Study

**Repository**: `sports-factuality-indic` (Evaluation of General Multilingual News Summarization)  
**Date**: September 6, 2026  
**Audited Artifacts**:
- `outputs/evaluation_manifest.jsonl` / `outputs/evaluation_manifest.csv` (Canonical Manifest, $N = 4,000$)
- `outputs/llm_judge_results.jsonl` (Audited Factuality Sample, $N = 320$ sampled, $N = 314$ parsed)
- `outputs/generation_validity_table.csv`
- `outputs/error_taxonomy_table.csv`
- `outputs/correlation_table.csv`
- `outputs/regression_results.txt` / `outputs/regression_table.csv`
- `outputs/COMPREHENSIVE_FACTUALITY_ANALYSIS_REPORT.md`

---

## 1. Denominators and Sample Sizes

| Dataset / Table | Denominator ($N$) | Description & Inclusion Rule |
| :--- | :---: | :--- |
| **Source Benchmark Pool** | **1,500** articles | 500 English, 500 Hindi, 500 Bengali articles sampled deterministically from XL-Sum test splits (`data/sports_filtered/{code}_test.jsonl`). |
| **Total Evaluated Manifest** | **4,000** summaries | 8 system $\times$ language conditions $\times$ 500 test articles per condition. |
| **Output-Level Validity Table** | **4,000** summaries | Full benchmark breakdown into `valid` (3,488), `malformed` (371), `truncated` (145), `wrong_script` (0), `empty` (0), `generation_error` (0). |
| **Surface Overlap Table** | **4,000** summaries | 500 summaries per system-language cell for ROUGE-1/2/L, chrF, and BERTScore-F1. |
| **LLM Judge Sampled Subset** | **320** summaries | 40 summaries sampled per system-language condition ($40 \times 8 = 320$). |
| **LLM Judge Verified Verdicts** | **314** summaries | Complete parsed verdicts from `openai/gpt-5.6-luna`. 6 items dropped due to API parse timeouts / rate limits (1 `indicbart/hi`, 1 `mt5_xlsum/bn`, 3 `mt5_xlsum/en`, 1 `mt5_xlsum/hi`). |
| **Summary-Level Correlation Table** | **314** (Judged) | Strict 1-row-per-summary Spearman $\rho$ and Pearson $r$ within each isolated condition. |
| **Multivariable OLS Regression** | **314** (Judged) | Dependent variable: factuality label ($0, 1, 2$). Degrees of freedom = 306. $R^2 = 0.4396$. |

---

## 2. Filtering and Preprocessing Rules

1. **Domain Selection**:
   - Initial screening for domain-specific sports keywords revealed $<5\%$ retention across Indic test splits. The benchmark scope was broadened to **General Multilingual News** (governance, politics, economics, crime, international news) to preserve sample size integrity without synthetic bias.
2. **Text Cleaning & Special Token Handling**:
   - Special tokens (`<s>`, `</s>`, `<2bn>`, `<2hi>`, `<pad>`, `[CLS]`, `[SEP]`, etc.) are preserved in `generated_summary` (raw model output) and stripped in `cleaned_summary` exclusively for metric computation.
   - Unicode NFC normalization is applied across all strings.
3. **Indic Tokenization (`IndicMultilingualTokenizer`)**:
   - Standard `rouge_score` strips non-ASCII text. The custom tokenizer splits on whitespace while isolating punctuation marks (including Devanagari `।`, `॥`) without discarding Devanagari or Bengali script characters.
4. **chrF & BERTScore Configuration**:
   - chrF is evaluated via `sacrebleu` on a standard 0.0–100.0 scale.
   - BERTScore uses `microsoft/deberta-v3-base` for English and `google/muril-base-cased` for Hindi and Bengali.

---

## 3. Label Definitions & Taxonomy

### A. Output-Level Validity (N = 4,000)
- `valid`: Coherent, readable natural language string in the expected script.
- `malformed`: Special-token-only outputs, repetitive degenerate n-grams (e.g., `-------`, `ख़ ख़ ख़`), or tokenizer corruption artifacts.
- `wrong_script`: Text generated in a completely different script than target (e.g., Devanagari in Bengali context).
- `truncated`: Unfinished fragments shorter than 10 characters or 3 words.
- `empty`: Empty string or whitespace only.
- `generation_error`: Uncaught runtime generation exception.

### B. Summary-Level Factuality Labels (N = 314)
Derived via `openai/gpt-5.6-luna` on an ordinal scale:
- `2` (**Fully Factual**): All assertions in the summary are completely supported by the source document.
- `1` (**Minor Distortion / Omission / Partial**): The summary is mostly faithful but omits key qualifying context or contains non-critical minor inaccuracies.
- `0` (**Severe Contradiction / Hallucination / Degenerate**): The summary contains direct factual contradictions, invented entity relations, false numerical assertions, or is uninterpretable gibberish.

### C. Claim-Level Error Subtypes
- `person_identity`: Incorrectly attributing actions to the wrong person or inventing names.
- `organization_identity`: Distorting or inventing company/governmental affiliations.
- `location_entity`: Swapping geographic places or associating actions with the wrong city/country.
- `event_relation`: Inverting cause-and-effect, reversing winners/losers, or misrepresenting actions between entities.
- `numeric_fact`: Altering quantities, casualty counts, scores, or measurements.
- `temporal_fact`: Distorting dates, years, or sequence of events.
- `unsupported_claim`: Asserting claims with zero source evidence.
- `omission`: Leaving out critical core facts.
- `malformed_output`: Degenerate or uninterpretable character loop.

---

## 4. Systems and Checkpoints Evaluated

| System Key | HuggingFace Checkpoint / Source | Architecture / Method | Decoding Parameters |
| :--- | :--- | :--- | :--- |
| `lead_baseline` | Extractive lead sentence(s) | Heuristic | First 1–2 sentences matching reference length |
| `indicbart` | `ai4bharat/IndicBARTSS` | Seq2Seq Transformer (Pretrained Indic) | Beam size 4, length penalty 1.0, max length 128 |
| `mt5_xlsum` | `csebuetnlp/mT5_multilingual_XLSum` | Multilingual T5 (300M params, fine-tuned on XL-Sum) | Prefix `summarize: `, beam size 4, length penalty 1.0, max length 128 |

---

## 5. Known Limitations & Audit Notes

1. **IndicBART Bengali Root Cause**: IndicBART in Bengali failed due to omission of `forced_bos_token_id=tokenizer.encode('<2bn>')[0]` during generation. This is an implementation/configuration compatibility failure, not a linguistic finding about Bengali.
2. **LLM Judge Proxy**: LLM factuality annotations are automatic judge-derived labels, not human ground truth.
3. **Non-Causal Diagnostic Scope**: Statistical associations (correlations, OLS regression) reflect observational characteristics of the benchmark and should not be interpreted as causal claims.
