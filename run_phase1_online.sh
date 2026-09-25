#!/usr/bin/env bash
# Run this on the CPU LOGIN NODE (must have internet access).
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Phase 1: downloading dataset + all model weights ==="
python scripts/01_download_all.py --config configs/pipeline.yaml

echo "=== Phase 1 complete ==="
echo "Next: copy/rsync this directory to wherever the GPU node reads from"
echo "(or confirm it's already on a shared filesystem), then run:"
echo "  bash run_phase2_offline.sh"
