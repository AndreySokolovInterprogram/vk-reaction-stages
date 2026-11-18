#!/usr/bin/env bash
# Feed features u_p (GPU required). Each stage resumes from its last checkpoint.
#   1) Qwen3-VL-Embedding-2B embeddings for all posts;
#   2) waits for the flag $MEDIA_ROOT/out/VLM_GO (content = batch size; set after a speed test and a visual check
#      of VLM answers);
#   3) predefined triggers with Qwen3-VL-4B-Instruct (4-bit) for all posts.
# Paths come from config.env in the repository root (MEDIA_ROOT, PYTHON_MEDIA).
# Input: $MEDIA_ROOT/data/media_items.parquet (build_media_items.py). Progress: $MEDIA_ROOT/logs/progress.log.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${CONFIG:-$(dirname "$HERE")/config.env}"
if [ -f "$CFG" ]; then set -a; . "$CFG"; set +a; fi
: "${MEDIA_ROOT:?set MEDIA_ROOT in config.env}"
export MEDIA_ROOT
PY="${PYTHON_MEDIA:-python3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$MEDIA_ROOT/logs" "$MEDIA_ROOT/out"
cd "$MEDIA_ROOT"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
if [ ! -f out/embed/DONE ]; then
  echo "$(ts) pipeline: embed" | tee -a logs/progress.log
  "$PY" -u "$HERE/features.py" embed --batch 4
fi
while [ ! -f out/VLM_GO ]; do sleep 60; done
if [ ! -f out/vlm/DONE ]; then
  echo "$(ts) pipeline: vlm (batch $(cat out/VLM_GO))" | tee -a logs/progress.log
  "$PY" -u "$HERE/features.py" vlm --batch "$(cat out/VLM_GO)"
fi
echo "$(ts) pipeline: done" | tee -a logs/progress.log
