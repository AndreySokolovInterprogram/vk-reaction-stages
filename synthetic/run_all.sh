#!/usr/bin/env bash
# Synthetic identifiability check of the multistage model (Sec. V-E of the paper).
# Data are generated inside the scripts (fixed seeds); no input needed. Under a minute on a CPU.
#   run1 - S1: Fisher information of the full model (SEs at icc 0 / 0.002 / 0.01)
#   run2 - S2-S3: a feature allowed in several stages; reposts only
#   run3 - S4-S5: missing e_C^- path; post-level heterogeneity in the feed stage, merged T and O
#   run4 - S6: wrong link function; S3 (reposts only) with fitting
#   run5 - merged T+O without heterogeneity; 10 replicates of S1 under heterogeneity sd 1
#   run6_real_alpha - S1, S2d at the real overdispersion (alpha from NB2); reconstruction, see its docstring
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
OUT="${SYNTH_OUT:-$HERE/out}"; mkdir -p "$OUT"
for i in 1 2 3 4 5 6_real_alpha; do echo "== run$i  $(date '+%H:%M:%S')"; "$PY" "$HERE/run$i.py" | tee "$OUT/run$i.txt"; done
