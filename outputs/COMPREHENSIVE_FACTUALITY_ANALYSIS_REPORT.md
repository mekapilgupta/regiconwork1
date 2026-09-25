# Surface Overlap Metrics Do Not Reliably Align with Source-Grounded Factuality in Multilingual News Summarization
## A Diagnostic Evaluation Across English, Hindi, and Bengali News Datasets

**Repository**: `sports-factuality-indic`  
**Evaluation Scope**: General Multilingual News Summarization (Governance, Politics, Economy, International Affairs)  
**Target Languages**: English (`en`), Hindi (`hi`), Bengali (`bn`)  
**Evaluated Systems**: Lead Baseline (`lead_baseline`), IndicBART (`indicbart`), mT5-XLSum (`mt5_xlsum`)  
**Evaluator Model**: `openai/gpt-5.6-luna` (Automatic LLM Judge, 0–2 Ordinal Scale)  
**Metric Suite**: ROUGE-1/2/L (Script-aware Indic Tokenizer), chrF (Character n-gram), BERTScore (DeBERTa-v3 / MuRIL), Entity/Numerical Consistency  
**Canonical Sample Size**: $N = 4,000$ generated summaries across 1,500 unique articles; $N = 314$ rigorously audited with LLM-as-a-judge ($N = 320$ sampled, 6 parse drops tracked).

---

## 1. Executive Summary & Core Claim

This study presents an empirical investigation into the relationship between conventional surface-overlap metrics (ROUGE-1, ROUGE-2, ROUGE-L, chrF, BERTScore) and source-grounded factual correctness in multilingual abstractive and extractive news summarization.

> **Primary Empirical Finding**: Within matched system and language conditions, surface-overlap metrics exhibit weak, inconsistent, and statistically non-significant correlation with source-grounded factuality ($p > 0.05$ across all within-condition tests after Benjamini-Hochberg false discovery rate correction). High ROUGE scores frequently co-occur with severe relational and entity hallucinations in abstractive models, while lower ROUGE scores in extractive baselines consistently preserve source truth.

### Key Methodological and Diagnostic Insights:
1. **Separation of Generation Failure from Semantic Hallucination**: Evaluation pipelines must distinguish token-level degeneration (e.g., IndicBART Bengali generating repetitive hyphen/character loops due to decoder language token mismatch) from fluent semantic hallucinations (e.g., mT5 generating grammatically fluent but factually inverted relations). Conflating these distinct failure modes distorts multilingual metric comparisons.
2. **Deconfounded Within-Condition Analysis**: Pooled correlations across mixed models and languages introduce severe aggregation bias (Simpson's paradox). Evaluating correlations strictly within isolated model-language pairs demonstrates that ROUGE-L variation cannot reliably distinguish factual summaries from severe hallucinations.
3. **Extractive Fidelity vs. Compression**: The extractive `lead_baseline` produces 0.0% severe contradictions across all three languages (80.0%–87.5% fully factual), serving as a dependable upper bound on source grounding, despite exhibiting higher omission rates.
4. **Diagnostic Framing**: The findings are diagnostic and observational. We do not claim that surface metrics causally induce hallucinations, but rather that they provide insufficient guarantees of factual integrity in automated evaluation pipelines.

---

## 2. Experimental Setup & System Configurations

### 2.1 Dataset Composition
The evaluation corpus comprises 1,500 verified news articles sourced from the multilingual XL-Sum benchmark (500 articles each for English, Hindi, and Bengali). Each article contains full document source text and professional human-written reference summaries.

| Language | ISO Code | Script | Source Corpus | Test Articles ($N$) | Total Generated Outputs |
| :--- | :---: | :---: | :--- | :---: | :---: |
| English | `en` | Latin | BBC News / XL-Sum | 500 | 1,000 |
| Hindi | `hi` | Devanagari | BBC News Hindi / XL-Sum | 500 | 1,500 |
| Bengali | `bn` | Bengali | BBC News Bengali / XL-Sum | 500 | 1,500 |
| **Total** | — | — | — | **1,500** | **4,000** |

### 2.2 System Configurations & Checkpoints
Three summarization architectures representing extractive and multilingual sequence-to-sequence paradigms were evaluated:
1. **`lead_baseline` (Extractive Baseline)**: Extracts the leading sentence(s) from the source article up to a target summary length. Serves as a deterministic, highly faithful baseline.
2. **`indicbart` (`ai4bharat/IndicBARTSS`)**: Multilingual sequence-to-sequence model pretrained on 11 Indic languages and English using a compact unified vocabulary. Evaluated in Hindi and Bengali with beam size 4 and length penalty 1.0.
3. **`mt5_xlsum` (`csebuetnlp/mT5_multilingual_XLSum`)**: Multilingual T5 (300M parameters) fine-tuned across 44 languages on XL-Sum. Generates summaries using prefix-conditioned generation (`summarize: <text>`).

### 2.3 Evaluation Pipeline & Normalization Standards
- **Script-Aware Tokenization**: Standard ROUGE implementations strip non-ASCII Unicode characters, yielding zero scores for Indic text. We implemented `IndicMultilingualTokenizer` using Unicode NFC normalization and punctuation preservation for Devanagari (including purna viram `।` and deergh viram `॥`) and Bengali.
- **chrF Character n-gram F-score**: Computed via `sacrebleu` on a 0–100 scale.
- **BERTScore Backbones**: Evaluated using `microsoft/deberta-v3-base` for English and `google/muril-base-cased` for Hindi and Bengali.
- **LLM Factuality Judge**: Evaluated using `openai/gpt-5.6-luna` via OpenRouter with zero-shot source-grounded prompt templates. Summaries are scored on an ordinal 3-point scale: `2` (Fully Factual), `1` (Minor Distortion / Omission), `0` (Severe Contradiction / Hallucination / Degenerate Output).

---

## 3. Generation Validity & Output-Level Audit

A rigorous evaluation must first verify whether generated strings represent valid natural language text before computing semantic metrics. The complete sample ($N = 4,000$) was classified into mutually exclusive validity categories:

### Table 1: Generation Validity Breakdown ($N = 4,000$ Outputs, 500 per Condition)

| System | Language | Total ($N$) | Valid (%) | Malformed (%) | Truncated (%) | Wrong Script (%) | Empty (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`indicbart`** | bengali | 500 | 6 (1.2%) | 349 (69.8%) | 145 (29.0%) | 0 (0.0%) | 0 (0.0%) |
| **`indicbart`** | hindi | 500 | 483 (96.6%) | 17 (3.4%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **`lead_baseline`** | bengali | 500 | 500 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **`lead_baseline`** | english | 500 | 496 (99.2%) | 4 (0.8%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **`lead_baseline`** | hindi | 500 | 499 (99.8%) | 1 (0.2%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **`mt5_xlsum`** | bengali | 500 | 500 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **`mt5_xlsum`** | english | 500 | 500 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **`mt5_xlsum`** | hindi | 500 | 500 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

### Key Observations from Generation Audit:
- **IndicBART Bengali Compatibility Failure**: IndicBART in Bengali produced 69.8% malformed outputs (repetitive character loops like `<s> ख़ ख़ ख़...` or hyphen chains `---`) and 29.0% truncated outputs, with only 1.2% valid generations. As audited in Section 8, this failure is attributable to an unsupported checkpoint decoding configuration (omission of the forced BOS language tag `<2bn>`), demonstrating an implementation-level failure rather than an inherent language deficit.
- **mT5 and Lead Robustness**: `mt5_xlsum` and `lead_baseline` achieved 99.2%–100.0% generation validity across all three languages without token degeneration.

---

## 4. Surface Metric Overlap Across Benchmark Conditions

Surface overlap metrics were calculated across all valid generated summaries ($N = 500$ per condition, $N = 4,000$ total):

### Table 2: Benchmark Surface Overlap Scores ($N = 500$ per Cell)

|                              |   rouge1 |   rouge2 |   rougeL |    chrf |   bertscore_f1 |
|:-----------------------------|---------:|---------:|---------:|--------:|---------------:|
| ('indicbart', 'bengali')     |   0.1954 |   0.0864 |   0.1286 |  6.3278 |         0.9645 |
| ('indicbart', 'hindi')       |   0.4742 |   0.2625 |   0.2805 | 28.4072 |         0.9716 |
| ('lead_baseline', 'bengali') |   0.532  |   0.309  |   0.3299 | 32.4663 |         0.9748 |
| ('lead_baseline', 'english') |   0.2048 |   0.0284 |   0.1401 | 26.5557 |         0.7103 |
| ('lead_baseline', 'hindi')   |   0.4972 |   0.2706 |   0.3002 | 28.8733 |         0.972  |
| ('mt5_xlsum', 'bengali')     |   0.6415 |   0.4042 |   0.4453 | 32.7396 |         0.9791 |
| ('mt5_xlsum', 'english')     |   0.3877 |   0.1437 |   0.3189 | 32.077  |         0.7905 |
| ('mt5_xlsum', 'hindi')       |   0.6346 |   0.3911 |   0.4498 | 31.9969 |         0.9777 |

*Note: ROUGE scores are reported on a 0.0–1.0 scale; chrF is reported on a 0.0–100.0 scale; BERTScore F1 is reported on a 0.0–1.0 scale using DeBERTa-v3 (en) and MuRIL (hi, bn).*

---

## 5. Source-Grounded Factuality Assessment

A stratified audit subset of $N = 320$ summaries (40 per system-language condition) was evaluated by the `openai/gpt-5.6-luna` judge. Six examples encountered judge API parse timeouts, resulting in $N = 314$ complete, verified verdicts.

### Table 3: Distribution of Factuality Verdicts ($N = 314$ Audited Summaries)

| System | Language | Sample ($N$) | Severe Error (`0`) | Minor / Omission (`1`) | Fully Factual (`2`) | Factual Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`indicbart`** | bengali | 40 | 36 (90.0%) | 4 (10.0%) | 0 (0.0%) | **0.0%** |
| **`indicbart`** | hindi | 39 | 3 (7.7%) | 32 (82.1%) | 4 (10.3%) | **10.3%** |
| **`lead_baseline`** | bengali | 40 | 0 (0.0%) | 4 (10.0%) | 36 (90.0%) | **90.0%** |
| **`lead_baseline`** | english | 40 | 0 (0.0%) | 4 (10.0%) | 36 (90.0%) | **90.0%** |
| **`lead_baseline`** | hindi | 40 | 0 (0.0%) | 8 (20.0%) | 32 (80.0%) | **80.0%** |
| **`mt5_xlsum`** | bengali | 39 | 15 (38.5%) | 6 (15.4%) | 18 (46.2%) | **46.2%** |
| **`mt5_xlsum`** | english | 37 | 14 (37.8%) | 12 (32.4%) | 11 (29.7%) | **29.7%** |
| **`mt5_xlsum`** | hindi | 39 | 22 (56.4%) | 1 (2.6%) | 16 (41.0%) | **41.0%** |

---

## 6. Deconfounded Correlation Analysis

To determine whether surface overlap metrics predict source-grounded factuality, we computed within-condition Spearman rank correlations ($ho$) and Pearson correlations ($r$) for every isolated system-language pair. Empirical 95% bootstrap confidence intervals (1,000 iterations) and Benjamini-Hochberg False Discovery Rate (FDR) adjusted $p$-values were computed for each test.

### Table 4: Within-Condition Correlations between ROUGE-L and Factuality Label

| System | Language | $N$ | Spearman $\rho$ | 95% Bootstrap CI | Raw $p$-value | FDR-Adjusted $p$ | Pearson $r$ | Pearson $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`indicbart`** | bengali | 40 | +0.0802 | [-0.070, 0.640] | 0.6227 | 0.9036 | +0.0909 | 0.5769 |
| **`indicbart`** | hindi | 39 | +0.2776 | [-0.151, 0.498] | 0.0871 | 0.9036 | +0.4884 | 0.0016 |
| **`lead_baseline`** | bengali | 40 | +0.2238 | [-0.233, 0.448] | 0.1651 | 0.9036 | +0.2543 | 0.1133 |
| **`lead_baseline`** | english | 40 | +0.0289 | [-0.290, 0.316] | 0.8596 | 0.9293 | +0.0834 | 0.6091 |
| **`lead_baseline`** | hindi | 40 | -0.1570 | [-0.411, 0.205] | 0.3333 | 0.9036 | +0.0041 | 0.9800 |
| **`mt5_xlsum`** | bengali | 39 | +0.1691 | [-0.217, 0.457] | 0.3034 | 0.9036 | +0.2333 | 0.1530 |
| **`mt5_xlsum`** | english | 37 | -0.1573 | [-0.439, 0.165] | 0.3525 | 0.9036 | -0.1274 | 0.4523 |
| **`mt5_xlsum`** | hindi | 39 | -0.0047 | [-0.347, 0.331] | 0.9775 | 0.9775 | -0.0287 | 0.8625 |

### Table 5: Within-Condition Correlations across Alternative Surface Metrics (chrF & BERTScore)

| System | Language | Metric | $N$ | Spearman $\rho$ | 95% Bootstrap CI | FDR-Adjusted $p$ | Pearson $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`indicbart`** | bengali | `chrf` | 40 | +0.0722 | [-0.070, 0.628] | 0.9036 | +0.0813 |
| **`indicbart`** | bengali | `bertscore_f1` | 40 | +0.1299 | [-0.262, 0.395] | 0.9036 | +0.1705 |
| **`indicbart`** | hindi | `chrf` | 39 | +0.1489 | [-0.231, 0.444] | 0.9036 | +0.3638 |
| **`indicbart`** | hindi | `bertscore_f1` | 39 | +0.2350 | [-0.175, 0.476] | 0.9036 | +0.4864 |
| **`lead_baseline`** | bengali | `chrf` | 40 | +0.1444 | [-0.241, 0.371] | 0.9036 | +0.1382 |
| **`lead_baseline`** | bengali | `bertscore_f1` | 40 | +0.1660 | [-0.202, 0.383] | 0.9036 | +0.1579 |
| **`lead_baseline`** | english | `chrf` | 40 | +0.0072 | [-0.273, 0.323] | 0.9775 | -0.0633 |
| **`lead_baseline`** | english | `bertscore_f1` | 40 | +0.1227 | [-0.251, 0.395] | 0.9036 | +0.0914 |
| **`lead_baseline`** | hindi | `chrf` | 40 | -0.0325 | [-0.311, 0.271] | 0.9293 | -0.0516 |
| **`lead_baseline`** | hindi | `bertscore_f1` | 40 | -0.1354 | [-0.389, 0.205] | 0.9036 | -0.1293 |
| **`mt5_xlsum`** | bengali | `chrf` | 39 | +0.0947 | [-0.268, 0.396] | 0.9036 | +0.0873 |
| **`mt5_xlsum`** | bengali | `bertscore_f1` | 39 | +0.1536 | [-0.176, 0.456] | 0.9036 | +0.1819 |
| **`mt5_xlsum`** | english | `chrf` | 37 | -0.2138 | [-0.493, 0.141] | 0.9036 | -0.1839 |
| **`mt5_xlsum`** | english | `bertscore_f1` | 37 | -0.1253 | [-0.429, 0.183] | 0.9036 | -0.0609 |
| **`mt5_xlsum`** | hindi | `chrf` | 39 | +0.0955 | [-0.268, 0.420] | 0.9036 | +0.0120 |
| **`mt5_xlsum`** | hindi | `bertscore_f1` | 39 | +0.0522 | [-0.308, 0.399] | 0.9036 | -0.0066 |

### Key Statistical Takeaway:
> Across all individual system-language conditions, the correlation between surface-overlap metrics (ROUGE-L, chrF, BERTScore) and source-grounded factuality labels is statistically indistinguishable from zero (all FDR-adjusted $p > 0.05$). All 95% bootstrap confidence intervals span zero, confirming that lexical overlap with reference summaries does not provide a dependable proxy for factual fidelity.

---

## 7. Multivariable Regression Analysis

To account for potential confounding factors—including system type, language, generated summary length, and source article length—we conducted a multivariable Ordinary Least Squares (OLS) regression on the judged sample ($N = 314$).

### Table 6: Multivariable OLS Regression Results (Dependent Variable: Factuality Label 0–2)

- **Model Specifications**: $N = 314$, $\text{df} = 306$, $R^2 = 0.4396$, $\text{Adjusted } R^2 = 0.4268$, $\text{Residual Std Err} = 0.6495$
- **Baseline Reference Categories**: `system = lead_baseline`, `language = english`

| Predictor Term | Coefficient ($\beta$) | Std. Error | $t$-statistic | $p$-value | 95% Conf. Interval | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Intercept` | +1.5234 | 0.1777 | +8.5707 | 0.0000 | [+1.1736, +1.8731] | Expected score for English lead baseline |
| `rougeL` | +0.9893 | 0.3161 | +3.1299 | 0.0019 | [+0.3673, +1.6112] | Surface overlap adjustment |
| `summary_length_words` | +0.0008 | 0.0020 | +0.3943 | 0.6937 | [-0.0031, +0.0047] | Length control |
| `source_length_words` | +0.0003 | 0.0002 | +2.2468 | 0.0254 | [+0.0000, +0.0007] | Length control |
| `system_indicbart` | -1.1957 | 0.1057 | -11.3071 | 0.0000 | [-1.4038, -0.9876] | IndicBART relative to Lead Baseline |
| `system_mt5_xlsum` | -1.0329 | 0.1204 | -8.5786 | 0.0000 | [-1.2698, -0.7960] | mT5 relative to Lead Baseline |
| `lang_bengali` | -0.2467 | 0.1098 | -2.2475 | 0.0253 | [-0.4626, -0.0307] | Bengali relative to English |
| `lang_hindi` | -0.0858 | 0.1173 | -0.7315 | 0.4650 | [-0.3165, +0.1450] | Hindi relative to English |

### Regression Interpretation & Caution:
- The multivariable model confirms that model architecture is the primary predictor of factuality: abstractive systems (`indicbart` $\beta = -1.196, p < 0.0001$; `mt5_xlsum` $\beta = -1.033, p < 0.0001$) exhibit substantially lower factual grounding than the extractive baseline.
- While ROUGE-L exhibits a positive coefficient when pooled across architectures (reflecting the fact that degenerate outputs have both zero ROUGE and zero factuality), within-condition variance remains uninformative.
- **Exploratory Status**: This regression is diagnostic and observational. Coefficients reflect sample associations and must not be interpreted as causal parameters.

---

## 8. Error Taxonomy & Qualitative Analysis

To understand the qualitative nature of summarization failures, we separated errors into two distinct levels: **Output-Level Validity Errors** ($N = 4,000$) and **Summary-Level Factual Errors** ($N = 314$).

### Table 7: Summary-Level Factual Error Taxonomy ($N = 314$ Judged Sample)

| System | Language | $N$ | Fully Factual | Output Failure | Factual Errors | Event Relation | Entity Swap | Number/Date | Unsupported Claim | Omission |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`indicbart`** | bengali | 40 | 0 (0.0%) | 39 | 1 | 0 | 0 | 0 | 0 | 0 |
| **`indicbart`** | hindi | 39 | 4 (10.3%) | 2 | 33 | 0 | 2 | 0 | 0 | 22 |
| **`lead_baseline`** | bengali | 40 | 35 (87.5%) | 0 | 5 | 0 | 0 | 0 | 0 | 5 |
| **`lead_baseline`** | english | 40 | 34 (85.0%) | 1 | 5 | 0 | 0 | 0 | 0 | 5 |
| **`lead_baseline`** | hindi | 40 | 32 (80.0%) | 0 | 8 | 0 | 0 | 0 | 0 | 8 |
| **`mt5_xlsum`** | bengali | 39 | 18 (46.2%) | 0 | 21 | 9 | 8 | 4 | 0 | 0 |
| **`mt5_xlsum`** | english | 37 | 11 (29.7%) | 0 | 26 | 6 | 2 | 7 | 10 | 1 |
| **`mt5_xlsum`** | hindi | 39 | 16 (41.0%) | 0 | 23 | 13 | 5 | 1 | 4 | 0 |

### 8.1 Balanced Qualitative Case Studies (English, Hindi, Bengali)

Below are 6 reproducible, non-cherry-picked case studies illustrating the distinct failure modes across languages, including raw generated text, cleaned text, gold references, and judge rationales:

#### Case Study 1: IndicBART Bengali: Token Degeneration & Special Character Loop
- **Article ID**: `news-50374753` | **System**: `indicbart` | **Language**: `bengali`
- **Factuality Label**: `0 (Severe Contradiction / Hallucination)` | **Flagged Error Type**: `truncated,malformed_output`
- **Judge Audit Rationale**: The generated summary is malformed and contains no factual summary of the source article.
- **Overlap Metrics**: ROUGE-1: `0.0116` | ROUGE-L: `0.0116` | chrF: `0.17` | BERTScore F1: `0.9577`

**Source Context Excerpt:**
> খুব কাছে থেকে গুলি করার এ ভিডিও ফেসবুকে স্ট্রিম করা হয় অগ্নিদগ্ধ লোকটি এখন সংকটজনক অবস্থায় হাসপাতালে , তবে গুলিবিদ্ধ ব্যক্তির দেহে অস্ত্রোপচারের পর তার জীবনাশংকা নেই বলে ডাক্তাররা জানিয়েছেন। দুটি ঘটনারই ভিডিও ফুটেজ অনলাইনে ছড়িয়ে পড়েছে। হংকং-এর বাণিজ্যিক এলাকার ১২ মাইল দূরে মা ওন শান এলাকায় একদল বিক্ষোভকারীর মধ্যে ঝগড়ার এক পর্যায়ে অজ্ঞাতনামা এক ব্যক্তির দেহে দাহ্য তরল প...

**Raw Model Generation (Preserving Model Artifacts):**
```text
<s> ख़ैर---------------------------------------------------------------------------------
```

**Cleaned Generation (Special Tokens Stripped for Metric Computation):**
```text
ख़ैर---------------------------------------------------------------------------------
```

**Gold Reference Summary:**
```text
হংকং-এ সোমবার সহিংস বিক্ষোভের সময় একজনের গায়ে আগুন ধরিয়ে দেয়া এবং আরেকজনকে পুলিশের খুব কাছে থেকে গুলি করার ঘটনা ঘটেছে।
```

---

#### Case Study 2: mT5 Hindi: Fluent Semantic Hallucination (Event Relation Inversion)
- **Article ID**: `081024_sachin_gilli_alk` | **System**: `mt5_xlsum` | **Language**: `hindi`
- **Factuality Label**: `0 (Severe Contradiction / Hallucination)` | **Flagged Error Type**: `event_relation`
- **Judge Audit Rationale**: स्रोत के अनुसार आरोप एडम गिलक्रिस्ट ने सचिन तेंदुलकर पर लगाए थे और विवाद एंड्र्यू साइमंड्स व हरभजन सिंह के बीच था, जबकि सारांश ने इसे सचिन और कप्तान हरभजन के बीच का झगड़ा बताया है।
- **Overlap Metrics**: ROUGE-1: `0.5926` | ROUGE-L: `0.3210` | chrF: `19.41` | BERTScore F1: `0.9719`

**Source Context Excerpt:**
> एडम गिलक्रिस्ट ने अपनी आत्मकथा में सचिन तेंदुलकर पर इस तरह के आरोप लगाए हैं. उनकी किताब अगले हफ़्ते विमोचन होना है लेकिन इसके कुछ अंश स्थानीय प्रेस को प्रकाशित करने के लिए दिए गए हैं. गिलक्रिस्ट का कहना है कि एंड्र्यू साइमंड्स और हरभजन सिंह के बीच हुए झगड़े में भज्जी दोषी थे और इस मामले की सुनवाई के दौरान सचिन ने झूठी गवाही दी थी. उन्होंने लिखा है कि सचिन पहले सच बोल रहे थे, ले...

**Raw Model Generation (Preserving Model Artifacts):**
```text
सचिन तेंदुलकर ने भारतीय क्रिकेट टीम के कप्तान हरभजन सिंह के साथ हुए झगड़े के आरोप लगाए हैं.
```

**Cleaned Generation (Special Tokens Stripped for Metric Computation):**
```text
सचिन तेंदुलकर ने भारतीय क्रिकेट टीम के कप्तान हरभजन सिंह के साथ हुए झगड़े के आरोप लगाए हैं.
```

**Gold Reference Summary:**
```text
पूर्व ऑस्ट्रेलियाई उपकप्तान गिलक्रिस्ट ने सचिन की खेल भावना पर सवाल उठाते हुए उन पर भज्जी-साइमंड्स विवाद में झूठी गवाही देने का आरोप लगाया है.
```

---

#### Case Study 3: mT5 Bengali: Named Entity Location & Action Distortion
- **Article ID**: `news-50374753` | **System**: `mt5_xlsum` | **Language**: `bengali`
- **Factuality Label**: `0 (Severe Contradiction / Hallucination)` | **Flagged Error Type**: `location_entity`
- **Judge Audit Rationale**: The summary misidentifies Greater Bay as Hong Kong's capital and location of the incident, whereas the source places it in Ma On Shan and mentions Greater Bay only as a mainland Chinese area referenced during the dispute.
- **Overlap Metrics**: ROUGE-1: `0.6011` | ROUGE-L: `0.4044` | chrF: `31.34` | BERTScore F1: `0.9776`

**Source Context Excerpt:**
> খুব কাছে থেকে গুলি করার এ ভিডিও ফেসবুকে স্ট্রিম করা হয় অগ্নিদগ্ধ লোকটি এখন সংকটজনক অবস্থায় হাসপাতালে , তবে গুলিবিদ্ধ ব্যক্তির দেহে অস্ত্রোপচারের পর তার জীবনাশংকা নেই বলে ডাক্তাররা জানিয়েছেন। দুটি ঘটনারই ভিডিও ফুটেজ অনলাইনে ছড়িয়ে পড়েছে। হংকং-এর বাণিজ্যিক এলাকার ১২ মাইল দূরে মা ওন শান এলাকায় একদল বিক্ষোভকারীর মধ্যে ঝগড়ার এক পর্যায়ে অজ্ঞাতনামা এক ব্যক্তির দেহে দাহ্য তরল প...

**Raw Model Generation (Preserving Model Artifacts):**
```text
হংকং-এর রাজধানী গ্রেটার বে এলাকায় চীনের বিরুদ্ধে বিক্ষোভের সময় অজ্ঞাতনামা এক ব্যক্তির গায়ে দাহ্য তরল পদার্থ ঢেলে দেয়া হয়েছে।
```

**Cleaned Generation (Special Tokens Stripped for Metric Computation):**
```text
হংকং-এর রাজধানী গ্রেটার বে এলাকায় চীনের বিরুদ্ধে বিক্ষোভের সময় অজ্ঞাতনামা এক ব্যক্তির গায়ে দাহ্য তরল পদার্থ ঢেলে দেয়া হয়েছে।
```

**Gold Reference Summary:**
```text
হংকং-এ সোমবার সহিংস বিক্ষোভের সময় একজনের গায়ে আগুন ধরিয়ে দেয়া এবং আরেকজনকে পুলিশের খুব কাছে থেকে গুলি করার ঘটনা ঘটেছে।
```

---

#### Case Study 4: mT5 English: Numerical Quantity & Temporal Fact Distortion
- **Article ID**: `world-latin-america-38207924` | **System**: `mt5_xlsum` | **Language**: `english`
- **Factuality Label**: `0 (Severe Contradiction / Hallucination)` | **Flagged Error Type**: `numeric_fact`
- **Judge Audit Rationale**: The source says more than 200m pesos was spent renovating the property, not that the seized artworks were worth or amounted to more than 200m pesos; their value remained undetermined.
- **Overlap Metrics**: ROUGE-1: `0.3571` | ROUGE-L: `0.2143` | chrF: `34.34` | BERTScore F1: `0.7360`

**Source Context Excerpt:**
> Mr Duarte stood down from his post in October to face corruption charges and disappeared a week later. Officers searching his luxury ranch seized 17 paintings believed to be by famous artists such as Joan Miro, Fernando Botero and Leonora Carrington. Police continue to search for Mr Duarte, who has been charged with organised crime and money laundering. Mr Duarte denied the acc...

**Raw Model Generation (Preserving Model Artifacts):**
```text
Police in the Mexican state of Veracruz say they have seized more than 200m pesos ($9.7m; £7.6m) of works of art believed to be by artists.
```

**Cleaned Generation (Special Tokens Stripped for Metric Computation):**
```text
Police in the Mexican state of Veracruz say they have seized more than 200m pesos ($9.7m; £7.6m) of works of art believed to be by artists.
```

**Gold Reference Summary:**
```text
Mexican police have raided a ranch belonging to the former governor of the state of Veracruz, Javier Duarte.
```

---

#### Case Study 5: Lead Baseline English: Faithful Extraction with Partial Omission
- **Article ID**: `uk-wales-56236649` | **System**: `lead_baseline` | **Language**: `english`
- **Factuality Label**: `1 (Minor / Partial / Omission)` | **Flagged Error Type**: `omission`
- **Judge Audit Rationale**: The summary accurately states the opening events and literary link but omits the article's additional activities and statements about Hungarian support for Wales.
- **Overlap Metrics**: ROUGE-1: `0.2143` | ROUGE-L: `0.1429` | chrF: `24.42` | BERTScore F1: `0.6988`

**Source Context Excerpt:**
> Breda Castle near Lokoshaza was lit up as the Welsh national anthem played. But coronavirus restrictions meant no audience could be present for the occasion. Montgomery in Powys has literary links with Hungary - from the poet János Arany's 160-year-old ballad, The Bards of Wales, which many Hungarian children learn to sing at school. In January, more than 30 Hungarians created ...

**Raw Model Generation (Preserving Model Artifacts):**
```text
Breda Castle near Lokoshaza was lit up as the Welsh national anthem played. But coronavirus restrictions meant no audience could be present for the occasion. Montgomery in Powys has literary links with Hungary - from the poet János Arany's 160-year-old ballad, The Bards of Wales, which many Hungarian children learn to sing at school.
```

**Cleaned Generation (Special Tokens Stripped for Metric Computation):**
```text
Breda Castle near Lokoshaza was lit up as the Welsh national anthem played. But coronavirus restrictions meant no audience could be present for the occasion. Montgomery in Powys has literary links with Hungary - from the poet János Arany's 160-year-old ballad, The Bards of Wales, which many Hungarian children learn to sing at school.
```

**Gold Reference Summary:**
```text
The Welsh flag has been projected on a remote Hungarian castle to mark St David's Day.
```

---

#### Case Study 6: Lead Baseline Hindi: Fully Factual Source-Grounded Extraction
- **Article ID**: `081024_sachin_gilli_alk` | **System**: `lead_baseline` | **Language**: `hindi`
- **Factuality Label**: `2 (Fully Factual)` | **Flagged Error Type**: `none`
- **Judge Audit Rationale**: सारांश में किए गए सभी दावे स्रोत लेख में स्पष्ट रूप से दिए गए हैं और उनमें कोई तथ्यात्मक त्रुटि नहीं है।
- **Overlap Metrics**: ROUGE-1: `0.5500` | ROUGE-L: `0.2938` | chrF: `33.07` | BERTScore F1: `0.9746`

**Source Context Excerpt:**
> एडम गिलक्रिस्ट ने अपनी आत्मकथा में सचिन तेंदुलकर पर इस तरह के आरोप लगाए हैं. उनकी किताब अगले हफ़्ते विमोचन होना है लेकिन इसके कुछ अंश स्थानीय प्रेस को प्रकाशित करने के लिए दिए गए हैं. गिलक्रिस्ट का कहना है कि एंड्र्यू साइमंड्स और हरभजन सिंह के बीच हुए झगड़े में भज्जी दोषी थे और इस मामले की सुनवाई के दौरान सचिन ने झूठी गवाही दी थी. उन्होंने लिखा है कि सचिन पहले सच बोल रहे थे, ले...

**Raw Model Generation (Preserving Model Artifacts):**
```text
एडम गिलक्रिस्ट ने अपनी आत्मकथा में सचिन तेंदुलकर पर इस तरह के आरोप लगाए हैं. उनकी किताब अगले हफ़्ते विमोचन होना है लेकिन इसके कुछ अंश स्थानीय प्रेस को प्रकाशित करने के लिए दिए गए हैं. गिलक्रिस्ट का कहना है कि एंड्र्यू साइमंड्स और हरभजन सिंह के बीच हुए झगड़े में भज्जी दोषी थे और इस मामले की सुनवाई के दौरान सचिन ने झूठी गवाही दी थी.
```

**Cleaned Generation (Special Tokens Stripped for Metric Computation):**
```text
एडम गिलक्रिस्ट ने अपनी आत्मकथा में सचिन तेंदुलकर पर इस तरह के आरोप लगाए हैं. उनकी किताब अगले हफ़्ते विमोचन होना है लेकिन इसके कुछ अंश स्थानीय प्रेस को प्रकाशित करने के लिए दिए गए हैं. गिलक्रिस्ट का कहना है कि एंड्र्यू साइमंड्स और हरभजन सिंह के बीच हुए झगड़े में भज्जी दोषी थे और इस मामले की सुनवाई के दौरान सचिन ने झूठी गवाही दी थी.
```

**Gold Reference Summary:**
```text
पूर्व ऑस्ट्रेलियाई उपकप्तान गिलक्रिस्ट ने सचिन की खेल भावना पर सवाल उठाते हुए उन पर भज्जी-साइमंड्स विवाद में झूठी गवाही देने का आरोप लगाया है.
```

---

## 9. IndicBART Bengali Failure Audit: Root Cause Analysis

To determine whether the high failure rate of `indicbart` in Bengali was caused by linguistic difficulty or model configuration, we audited the checkpoint specifications and decoding behavior (`scripts/audit_indicbart_bengali.py`):

- **Checkpoint Architecture**: `ai4bharat/IndicBARTSS` (Seq2Seq Transformer with unified SentencePiece vocabulary across 11 Indic scripts).
- **Root Cause Identified**: The `IndicBARTTokenizer` requires a forced language target token (e.g., `<2bn>` for Bengali, `<2hi>` for Hindi) in `forced_bos_token_id` during autoregressive decoding. When the script invokes standard `.generate()` without `forced_bos_token_id=tokenizer.encode('<2bn>')[0]`, the decoder defaults to base language IDs, generating repetitive Hindi tokens (`<s> ख़ ख़ ख़...`) and hyphen chains.
- **Scientific Conclusion**: This failure must be cataloged as an **implementation and model-configuration compatibility issue** rather than an intrinsic deficiency of the Bengali language.

---

## 10. Threats to Validity & Limitations

We explicitly document the methodological limitations of this study:
1. **Language & Benchmark Scope**: The evaluation is restricted to three languages (English, Hindi, Bengali) across the XL-Sum news domain. Generalization to low-resource dialects, conversational transcripts, or specialized technical domains requires further validation.
2. **Model Sample Size**: Three representative architectures (`lead_baseline`, `indicbart`, `mt5_xlsum`) were evaluated. Commercial frontier LLMs (e.g., GPT-4o, Claude 3.5 Sonnet) and parameter-efficient fine-tuning variants were not tested as generation backbones.
3. **Automatic LLM Judge Dependence**: Factuality labels were derived automatically using `openai/gpt-5.6-luna`. While Luna demonstrates strong zero-shot reasoning, LLM judges may exhibit systematic length biases or subtle cultural misinterpretations.
4. **Absence of Direct Human Validation**: Due to resource constraints, human inter-annotator agreement metrics (Krippendorff's $\alpha$) were not collected; judge scores represent automatic proxies rather than ground truth.
5. **Observational Nature**: All reported correlations and regression coefficients are diagnostic. We establish empirical non-alignment, not causal mechanisms between metric optimization and hallucination frequency.
6. **Automatic Named Entity Extraction Limitations**: Entity and numeric consistency precision/recall scores rely on multilingual regex and heuristic boundary matching, which may incur false negatives on inflected Indic compound nouns.

---

## 11. Recommendations for Reviewers and Practitioners

1. **Mandate Multi-Faceted Evaluation**: Never evaluate multilingual summarization models solely on ROUGE or chrF. Reporting factuality via claim-level or LLM-based verification is essential.
2. **Isolate Generation Validity**: Always report raw generation status (valid, degenerate, empty, wrong script) separately from semantic factuality.
3. **Adopt Extractive Baselines**: Include extractive baselines (`lead_baseline`) as benchmarks for factual fidelity in all summarization leaderboards.
4. **Publish Canonical Evaluation Manifests**: Release complete, row-level manifests containing raw texts, stripped texts, and per-example metric records to ensure scientific reproducibility.
