# Cross-Lingual Information Retrieval (CLIR) at Scale

This repository contains the complete research code, experimental pipeline, evaluation scripts, LaTeX paper sources, figures, TREC run files, and synthesis benchmark results for our paper on Cross-Lingual Information Retrieval (CLIR) over the 21M English Wikipedia passage collection (`wikipedia-dpr-100w`) and MS MARCO.

---

## 📁 Repository Structure

```
├── analysis/               # Diagnostic analysis, tail failure characterization, and headroom scripts
├── codefilesincon/         # Standalone source utilities & reference baseline code
├── config.yaml             # Main configuration file (budgets, models, evaluation parameters)
├── data/
│   ├── filtered/           # Evaluation topic splits (xor_bn, xor_te, miracl_hi, mmarco_hi)
│   ├── processed/          # LLM query translations & multi-view facet augmentations
│   └── raw/                # Query files and gold relevance judgments (qrels)
├── docs/                   # Benchmark skeletons and documentation
├── latexwork/              # Full camera-ready LaTeX sources, figures, bibliography & templates
├── metrics/                # Benchmark tables, stage CSVs, and statistical test outputs
├── plots/                  # Generated paper figures & risk-coverage comparison curves
├── report.md               # Complete synthesis report with full benchmark tables & CIs
├── reports/                # Validation reports
├── runs/                   # Plain-text TREC run evaluation files (Bengali & Telugu)
├── scripts/                # End-to-end stage execution & evaluation scripts (Stages 01-28)
├── src/                    # Core modules (retrieval, embeddings, LLM client, fusion, metrics)
├── state/                  # Checkpoint flags and execution metadata
├── third_party/            # Third-party benchmark evaluation scripts (official XOR evaluator)
├── tocodedetails.md        # Detailed architecture and pipeline documentation
└── .gitignore              # Configured to exclude heavy model checkpoints and large caches
```

---

## 🚀 Key Experimental Findings & Highlights

1. **Benchmark Datasets**: XOR-TyDi Development sets for Bengali (`bn`, n=312) and Telugu (`te`, n=238) against 21,015,324 passages, with Hindi (`hi`) mMARCO controls.
2. **Evaluated Pipelines**:
   - **Zero-Shot Dense (S1)** vs. **Translate-then-Search (S3)**
   - **ColBERT Multi-Vector & BM25 Sparse Channels**
   - **Reciprocal Rank Fusion (6-Channel Hybrid)**
   - **Cross-Lingual Reranking & Corpus Title Injection**
   - **Query Performance Prediction (QPP) Selective Escalation**
3. **Evaluation Protocol**: Official XOR-TyDi character-budget recall (R@2kt, R@5kt), R@100 candidate recall, Catastrophic Failure Rate (CFR), and 1,000x paired bootstrap confidence intervals with Holm-Bonferroni correction.

---

## 🛠️ Reproduction & Usage

### 1. Requirements
Ensure Python 3.10+ and standard NLP/IR dependencies are installed:
```bash
pip install torch transformers sentence-transformers pyserini nltk pandas numpy pyyaml scipy
```

### 2. Running Evaluation Scripts
Run the official token evaluation:
```bash
python scripts/27_evaluate_all_official_tokens.py
python scripts/28_evaluate_rq3_official_tokens.py
```

Run stages or paper figure generators:
```bash
python scripts/generate_paper_figures.py
```

---

## 📄 LaTeX Sources
All paper sources, plots, and formatting configurations are self-contained in `latexwork/`.
Compile with `pdflatex` or `lualatex`:
```bash
cd latexwork
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
