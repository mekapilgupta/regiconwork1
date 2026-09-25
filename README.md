# Sports Summarization Factuality in Indic Languages — Pipeline

Research question: do multilingual/Indic summarization models preserve sports
facts (winner, score, athlete names, dates, injury status), and do standard
metrics (ROUGE / BERTScore / chrF) detect the errors when they don't?

This repo is built for a **two-node university cluster**:

- **CPU login node** — has internet, no GPU. Used ONLY to download data,
  model weights, and tokenizers to local disk.
- **GPU node** — no internet (offline). Used to run inference, extract
  facts, and score everything, reading only from local disk.

A third stage uses the **OpenRouter API** (has internet, but we only touch
it from the CPU node or any node with outbound access to openrouter.ai) for
LLM-judge scoring. That stage is **hard-capped at $5.00** total spend by
`utils/budget_tracker.py`, which checks real cost after every call and
refuses to fire the next call if it would exceed the cap.

## Directory layout

```
configs/pipeline.yaml       <- single source of truth: langs, models, paths, budget
scripts/
  01_download_all.py        <- RUN ON CPU NODE (online). Downloads dataset + all HF models.
  02_filter_sports.py       <- CPU or GPU node (no internet needed). Filters XL-Sum to sports.
  03_run_summarization.py   <- RUN ON GPU NODE (offline). lead-3 baseline + mT5 + IndicBART.
  04_surface_metrics.py     <- GPU node (offline). ROUGE-1/2/L, chrF, BERTScore.
  05_fact_extraction.py     <- GPU node (offline). NER + regex fact-tuple extraction.
  06_fact_consistency.py    <- GPU node (offline). Compares source vs summary fact tuples.
  07_llm_judge.py            <- RUN ON CPU NODE or any node with internet. OpenRouter judge,
                                budget-capped at $5.
  08_analyze_results.py     <- Either node (offline, pandas/matplotlib only). Final tables + plots.
utils/
  io_utils.py                <- shared path/config/jsonl helpers
  budget_tracker.py          <- hard $ cap enforcement for OpenRouter calls
  openrouter_client.py       <- thin OpenRouter wrapper with real-cost polling
run_phase1_online.sh         <- convenience wrapper for the CPU/online steps
run_phase2_offline.sh        <- convenience wrapper for the GPU/offline steps
requirements.txt
```

## Step-by-step

### Phase 1 — CPU login node (online), run once

```bash
bash run_phase1_online.sh
```

This runs `01_download_all.py`, which:
1. Downloads the XL-Sum splits for the configured languages via `datasets`.
2. Downloads model weights for the summarizers (mT5-XLSum checkpoint,
   IndicBART) via `huggingface_hub.snapshot_download` (NOT `from_pretrained`,
   so nothing tries to hit the network again later).
3. Downloads NER models (IndicNER, spaCy `en_core_web_sm`) the same way.
4. Downloads the BERTScore backbone model weights.

Everything lands under `models/` and `data/` as plain files. After this
step finishes, `rsync`/`scp` (or shared filesystem, if CPU and GPU nodes
share `/home` or `/scratch`) the whole repo dir to wherever the GPU node
reads from. If CPU and GPU nodes share a filesystem (common on SLURM
clusters — check with your admin), you can skip the copy entirely.

Before moving to phase 2, set the offline env vars so nothing on the GPU
node can accidentally try to reach the internet and hang:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

(`run_phase2_offline.sh` sets these automatically.)

### Phase 2 — GPU node (offline)

```bash
bash run_phase2_offline.sh
```

Runs, in order: `02_filter_sports.py`, `03_run_summarization.py`,
`04_surface_metrics.py`, `05_fact_extraction.py`, `06_fact_consistency.py`.
Everything here reads/writes local disk only. No network calls. If any
library tries to phone home, the offline env vars above make it fail fast
instead of hanging on a dead connection.

### Phase 3 — LLM judge (needs internet + OpenRouter key)

Run from the CPU node (or any machine with outbound internet):

```bash
export OPENROUTER_API_KEY="sk-or-..."
python scripts/07_llm_judge.py --config configs/pipeline.yaml
```

This samples ~100 (source, generated-summary) pairs per language/system
(configurable), asks the judge model for an ordinal 0/1/2 factuality
verdict per fact type, and **stops itself** the instant cumulative real
spend would cross `$5.00`. It writes partial results incrementally, so a
budget-triggered stop never loses completed judgments.

### Phase 4 — Analysis

```bash
python scripts/08_analyze_results.py --config configs/pipeline.yaml
```

Produces `outputs/final_report.md`, a correlation table (ROUGE/BERTScore/chrF
vs. fact-tuple accuracy), per-language/per-system breakdowns, and the
confusion-style error typology (which fact types break most, by language).

## Cost control summary

- All embeddings, NER, translation-adjacent, and summarization models run
  **locally on the GPU node** — zero API cost.
- Only `07_llm_judge.py` touches a paid API, and only for: (a) judging
  factuality on a bounded sample, (b) optional query/paraphrase generation
  if you extend the pipeline. Both are metered against the same $5 cap.
- `utils/budget_tracker.py` persists spend to `outputs/budget_state.json`
  so the cap holds even across restarted/resumed runs.
