# CLIR Pilot — bn/te/hi → English Retrieval

Reproducible pilot comparing whole-query embedding, translation, and facet-wise
retrieval, on XOR-TyDi (bn/te → Wikipedia-DPR) and mMARCO (hi → MS MARCO),
with a monolingual control (Mr.TyDi / MIRACL).

## Why this repo is split into stages

Your cluster has two kinds of nodes:

| Node | Network | What it's for |
|---|---|---|
| **CPU node** | online | light work only: downloading data/models/indices, calling the OpenRouter LLM API (network I/O, not compute-heavy), deterministic leakage audits, JSON/report generation |
| **GPU node** | offline | heavy work only: loading the embedding model, encoding the candidate pool, dense retrieval, BM25 via Pyserini, fusion, bootstrap CI, metrics |

**Nothing in the GPU stage touches the network.** Everything it needs
(embedding model weights, Pyserini prebuilt indices, cached LLM outputs) must
already be on disk before you submit the GPU job. If a GPU-stage script tries
to hit the network it will raise `RuntimeError("network access attempted in
offline stage")` — see `src/config.py::assert_offline()`.

## Stage order

```
1. scripts/01_check_env.py        (CPU, online)  — sanity checks, GPU detect, LLM model-list ping
2. scripts/02_download_cpu_online.py (CPU, online)  — datasets, eval script, Pyserini indices, HF model weights
3. scripts/03_llm_stage_cpu_online.py (CPU, online)  — P_TRANSLATE / P_REWRITE / P_FACETS for all queries (async, cached)
   -- (leakage audit runs automatically at the end of stage 3, it's deterministic/cheap)
4. scripts/04_gpu_stage_offline.py  (GPU, OFFLINE) — embeddings, candidate pool, S1-S9 systems, ablations, metrics, bootstrap CI
5. scripts/05_judge_stage_cpu_online.py (CPU, online) — LLM judge silver labels on sampled queries (needs stage-4 run files)
6. scripts/06_report.py            (CPU, online or offline, no network needed) — report.md + plots
```

Every stage writes a `state/<stage>.done` marker per **track** (bn, te, hi,
hi_mono) and is safe to re-run — it will skip work that's already on disk
(cache hash files, existing run files, existing metrics).

## Test-run-first rule

Every stage takes `--n 100` (default) to run the 100-query pilot end to end
before you scale up. Only pass `--n -1` (full data) once the 100-query run
looks sane. This is enforced, not just a suggestion: stage 4 refuses to run
on `--n -1` unless `state/pilot100_ok.flag` exists (touched by
`scripts/06_report.py --mark-pilot-ok` after you've eyeballed the pilot report).

## Env file

Copy `.env.example` to `.env` and fill in:

```
LLM_PROVIDER=openrouter
LLM_MODEL=qwen/qwen-2.5-72b-instruct   # or google/gemini-... — set what you were given
LLM_API_KEY=sk-or-...
BUDGET_USD=5
```

`.env` is read once by `src/config.py`. The key is never logged or printed
anywhere (grep the codebase for `LLM_API_KEY` if you want to verify — it's
only ever passed as a header value).

## Directory layout (matches the spec)

```
data/        raw + filtered query sets (utf-8, NFC-normalized)
cache/llm/   sha256(model,prompt_version,input) -> json   (never re-called)
cache/emb/   passage embeddings, keyed by (encoder_revision, passage_id)
runs/        <system>__<track>.trec
metrics/     *.csv  (main, leak-free, ablations, strata, bootstrap CI)
analysis/    leakage.csv, hi_manual_review.csv, win/loss exports
plots/       per-query delta histogram, strata bar chart
state/       stage completion markers, resumability bookkeeping
config/      generated config.yaml snapshots (one per full run, timestamped)
```

## Models

- Embeddings: `BAAI/bge-m3` if GPU detected at stage 1, else
  `intfloat/multilingual-e5-small` (logged, and report is titled "CPU pilot").
- LLM: whatever `LLM_MODEL` resolves to via OpenRouter (you said qwen/gemini) —
  the model id is **never hard-coded** in source, only read from env and
  verified against the provider's `/models` list at stage 1.

## What this pilot cannot claim

Stated explicitly in `report.md` every run: pool-based recall (not full-corpus),
answer-containment qrels (not human relevance judgments) for XOR/Mr.TyDi/MIRACL
tracks, mMARCO uses graded qrels instead, single LLM used for all
translation/rewrite/facet/judge roles, judge labels are silver not ground
truth, Hindi manual review is by one Hindi-reading author (documented as
partial mitigation, not a substitute for independent annotation).
