#!/usr/bin/env bash
# All figures of the paper from the aggregated data in data/ (no raw data needed).
# Output: figures/out/sokol{1..8}_*.pdf (vector) + .png (600 dpi). Paths: FIGDATA_DIR, FIG_OUT (see common_style.py).
#   Figs. 1-3 schematics; 4 labelling accuracy; 5 scatter of shares; 6 accumulation; 7 effects; 8 invariance.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
cd "$HERE"
for s in make_sokol1_system.py make_sokol2_model.py make_sokol3_pipeline.py make_sokol4_audit.py \
         make_sokol5_shares.py make_sokol6_accumulation.py make_sokol7_effects.py make_sokol8_invariance.py; do
  echo "== $s"; "$PY" "$s"
done
