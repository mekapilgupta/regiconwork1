# Cross-Lingual Information Retrieval (CLIR) Pilot Architecture & Pipeline Guide

## 1. Overview & Problem Definition

This repository implements a rigorous, reproducible evaluation pipeline comparing:
1. **Whole-Query Multilingual Embedding**
2. **Translate-then-Search (Machine Translation baseline)**
3. **Structured Facet Extraction & Multi-Vector Hybrid Retrieval**

across three Indic languages:
- **Bengali (`bn`) & Telugu (`te`)**: XOR-TyDi dev queries evaluated against English Wikipedia passages using answer-containment metrics ($R@2\text{kt}$, $R@5\text{kt}$, Catastrophic Failure Rate).
- **Hindi (`hi`)**: mMARCO dev.small queries evaluated against MS MARCO passage corpus using official passage qrels (**MRR@10**, **Recall@20**, **Recall@100**, Catastrophic Failure Rate).
  > **Crucial Rule**: Hindi is kept in a **strictly separate track and separate table** (never merged with Bengali/Telugu numbers) because its queries come from mMARCO with passage qrels rather than XOR-TyDi answer spans.
- **Monolingual Controls (`hi_mono`, `bn_mono`, `te_mono`)**: MIRACL / Mr.TyDi monolingual retrieval ($q_{hi} \to p_{hi}$) to establish the "zero language gap" upper bound.

---

## 2. Server & Cluster Architecture: Online CPU vs. Offline GPU

To comply with the university cluster rules (preventing account suspension from heavy compute on the login node), the workflow is strictly decoupled into two node types:

| Environment | Connectivity | Allowed Workloads | Core Operations |
|---|---|---|---|
| **CPU Node** (Login / Interactive) | **Online** (Internet access) | Light I/O only (No heavy compute / no model inference) | Sanity checks, downloading datasets/model weights, calling OpenRouter LLM API (`httpx.AsyncClient`), deterministic leakage audit, report generation |
| **GPU Node** (Compute / SLURM) | **Offline** (No internet access) | Heavy compute & AI/ML tensor workloads | Loading pinned encoder weights (`BAAI/bge-m3`), encoding passage pools, Dense matrix search, BM25 indexing, Systems S1–S9 execution, bootstrap CI |

`assert_offline()` and `assert_online()` guards are enforced in code to prevent unauthorized network attempts or misplaced jobs.

---

## 3. Retrieval Systems (S1–S9) and Ablations

| System ID | Name | Query Representation / Retrieval Strategy |
|---|---|---|
| **S1** | Zero-Shot Dense | Original non-English query $q_{src}$ dense encoding against English passages |
| **S2** | Zero-Shot Sparse | BM25 lexical retrieval using original non-English query $q_{src}$ |
| **S3** | MT Dense | LLM English translation $T(q_{src})$ dense encoding |
| **S4** | Facet-Only Retrieval | Structured English entities, relation, and constraints ($F(q_{src})$) multi-vector search |
| **S5** | Facet + Translation Concat | Concatenation of $T(q_{src})$ and $F(q_{src})$ dense search |
| **S6** | **Full Proposed Hybrid Pipeline** | Reciprocal Rank Fusion (RRF) combining: MT Dense (S3) + Zero-shot Dense (S1) + Facet Dense (S4) + Facet BM25 (S4) + Translation BM25 (S9) |
| **S7** | Rewrite Control | Dense retrieval using LLM paraphrase $R(T(q_{src}))$ (controls for LLM rewording effects) |
| **S8** | Oracle English | Human English query (original English MS MARCO query for Hindi; human translation for XOR) |
| **S9** | Translation BM25 | BM25 sparse search using LLM translation $T(q_{src})$ |

### Ablation Studies:
- **A1 (`no_ent`)**: No entities (Relation + Constraints only)
- **A2 (`no_const`)**: No constraints (Entities + Relation only)
- **A3 (`no_rel`)**: No relations (Entities + Constraints only)
- **A4 (`surface_only`)**: Source language surface forms instead of English translated entities

---

## 4. Pipeline Execution Stages

### Stage 1: Environment & Sanity Checks (CPU, Online)
```bash
/home/kshitij22256/miniconda3/envs/slmtask/bin/python scripts/01_check_env.py
```
- Validates `.env` variables (`LLM_MODEL`, `LLM_API_KEY`, `BUDGET_USD`).
- Checks network connectivity to OpenRouter `/models`.
- Detects GPU / CUDA availability and pins the embedding model (`BAAI/bge-m3` on GPU, `intfloat/multilingual-e5-small` on CPU) to `state/embedding_model.json`.
- Freezes prompt versions (`translate_v1`, `rewrite_v1`, `facets_v1`).

### Stage 2: Download Datasets & HuggingFace Models (CPU, Online)
```bash
/home/kshitij22256/miniconda3/envs/slmtask/bin/python scripts/02_download_cpu_online.py --n 100
```
- Downloads XOR-TyDi (`bn`, `te`), mMARCO Hindi (`hi`), MS MARCO qrels & English queries, and MIRACL Hindi.
- Downloads HuggingFace model snapshot to `cache/hf/` for offline GPU use.
- Deterministically samples $N$ queries (default: 100 pilot, seed=42) to `data/filtered/`.

### Stage 3: LLM Generation & Deterministic Leakage Audit (CPU, Online)
```bash
/home/kshitij22256/miniconda3/envs/slmtask/bin/python scripts/03_llm_stage_cpu_online.py --concurrency 10
```
- Asynchronously calls OpenRouter API with bounded concurrency.
- Caches all LLM responses to `cache/llm/<sha256>.json` (never re-called).
- Validates extracted facet substrings against source questions.
- Runs deterministic leakage audit:
  - Exact & fuzzy answer containment for Bengali/Telugu.
  - 4+-token gold passage span overlap for Hindi.
- Exports `analysis/leakage.csv` and `analysis/hi_manual_review.csv` (100 Hindi queries for author review with empty verification columns).

### Stage 4: GPU Retrieval, System Execution & Metrics (GPU, Offline)
```bash
# Run on GPU node or via sbatch
/home/kshitij22256/miniconda3/envs/slmtask/bin/python scripts/04_gpu_stage_offline.py
```
- Runs 100% offline (`assert_offline()`).
- Encodes candidate passage pools and query representations.
- Runs S1–S9 and Ablations A1–A4 across all tracks.
- Exports TREC run files to `runs/<system>__<track>.trec`.
- Computes track-specific metrics (R@2kt/R@5kt for bn/te; MRR@10/Recall@20/Recall@100 for hi; nDCG@10 for mono).
- Calculates 95% Bootstrap Confidence Intervals and paired bootstrap significance testing ($p$-values against S6).

### Stage 5: Blinded LLM Judge Silver Labels (CPU, Online)
```bash
/home/kshitij22256/miniconda3/envs/slmtask/bin/python scripts/05_judge_stage_cpu_online.py --sample-queries 30
```
- Pools top passages across systems from stage-4 runs.
- Blinds system identities and prompts LLM judge for silver relevance labels (0–3), passage support, entity status, and constraint satisfaction.

### Stage 6: Report, Win/Loss Analysis & Plots (CPU, Online or Offline)
```bash
/home/kshitij22256/miniconda3/envs/slmtask/bin/python scripts/06_report.py --mark-pilot-ok
```
- Compiles `report.md` summarizing all tables, bootstrap CIs, and findings.
- Generates high-resolution performance plots in `plots/`.
- Exports 20 largest S6 wins and 20 largest S6 losses for Hindi with top-3 passages to `analysis/hi_win_loss_summary.csv`.
- Creates `state/pilot100_ok.flag` when `--mark-pilot-ok` is specified.

---

## 5. Directory Structure

```
├── .env                  # OpenRouter API key, model ID, budget configuration
├── .env.example          # Environment template
├── tocodedetails.md      # Comprehensive architecture and run documentation
├── report.md             # Auto-generated experiment report
├── src/                  # Core library modules
│   ├── config.py         # Settings, paths, offline/online guards, logger
│   ├── utils_io.py       # UTF-8/NFC normalization, JSONL I/O, sha256, stage markers
│   ├── prompts.py        # Versioned prompt templates and freeze checks
│   ├── llm_client.py     # Async OpenRouter client, budget check, disk caching
│   ├── facets.py         # Facet validation, string extraction, formatting
│   ├── leakage.py        # Answer containment & passage span leakage audits
│   ├── data_prep.py      # Dataset filtering & deterministic sampling (seed=42)
│   ├── download.py       # Idempotent raw data & HuggingFace model downloader
│   ├── embeddings.py     # Offline GPU/CPU dense encoder (bge-m3 / e5-small)
│   ├── retrieval.py      # BM25Index, DenseIndex, RRF, and Convex combination
│   ├── systems.py        # S1–S9 retrieval systems and Ablations A1–A4
│   └── metrics.py        # R@2kt, R@5kt, MRR@10, nDCG@10, bootstrap CI, p-values
├── scripts/              # 6 sequential pipeline entry points
│   ├── 01_check_env.py
│   ├── 02_download_cpu_online.py
│   ├── 03_llm_stage_cpu_online.py
│   ├── 04_gpu_stage_offline.py
│   ├── 05_judge_stage_cpu_online.py
│   └── 06_report.py
├── data/                 # Raw and filtered query sets
├── cache/                # LLM and embedding disk caches
├── runs/                 # TREC run files (<system>__<track>.trec)
├── metrics/              # Aggregated metric CSVs with Bootstrap 95% CIs
├── analysis/             # Leakage audit, Hindi manual review CSV, win/loss exports
├── plots/                # Generated performance figures
└── state/                # Completion markers and pinned model state
```
