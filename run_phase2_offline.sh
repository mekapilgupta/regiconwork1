#!/usr/bin/env bash
# Run this on the GPU NODE (no internet). Assumes data/ and models/ were
# already populated by run_phase1_online.sh and copied/shared onto this node.
set -euo pipefail

cd "$(dirname "$0")"

# Force every HF library to refuse network calls; fail fast instead of
# hanging on a dead connection if anything tries to reach the internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== Step 1/5: filter XL-Sum to sports articles ==="
python scripts/02_filter_sports.py --config configs/pipeline.yaml

echo "=== Step 2/5: run summarization systems ==="
python scripts/03_run_summarization.py --config configs/pipeline.yaml

echo "=== Step 3/5: compute surface metrics (ROUGE / BERTScore / chrF) ==="
python scripts/04_surface_metrics.py --config configs/pipeline.yaml

echo "=== Step 4/5: extract fact tuples (NER + regex) ==="
python scripts/05_fact_extraction.py --config configs/pipeline.yaml

echo "=== Step 5/5: compute fact consistency vs. source articles ==="
python scripts/06_fact_consistency.py --config configs/pipeline.yaml

echo "=== Phase 2 complete ==="
echo "Next (needs internet, run on CPU node or elsewhere):"
echo "  export OPENROUTER_API_KEY=sk-or-..."
echo "  python scripts/07_llm_judge.py --config configs/pipeline.yaml --dry_run   # check cost projection first"
echo "  python scripts/07_llm_judge.py --config configs/pipeline.yaml            # then run for real"
echo "Then, anywhere:"
echo "  python scripts/08_analyze_results.py --config configs/pipeline.yaml"
