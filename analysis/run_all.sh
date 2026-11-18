#!/usr/bin/env bash
# Main analysis of the VK panel (steps 1-3): pools, descriptive results, NB regressions, key_numbers.txt.
# Reads the export only. Paths come from config.env in the repository root (template: config.example.env)
# or from environment variables:
#   EXPORT_DIR   panel export directory (required)
#   VIDEO_VIEWS  CSV of main-video views of clips (optional)
#   TECH_GAP     CSV of posts affected by the collection gap (optional)
#   ANALYSIS_OUT results directory (default analysis/out)
#   COMMUNITY_ID if set, also profiles one community (33_community_profile.py)
#   PYTHON       interpreter (default python3 from the active venv)
# Usage:  ./analysis/run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
CFG="${CONFIG:-$ROOT/config.env}"
if [ -f "$CFG" ]; then set -a; . "$CFG"; set +a; fi
: "${EXPORT_DIR:?set EXPORT_DIR in config.env}"
EXPORT_DIR="$(cd "$EXPORT_DIR" && pwd)"
OUT_DIR="${ANALYSIS_OUT:-$HERE/out}"
VIDEO_VIEWS="${VIDEO_VIEWS:-}"
TECH_GAP="${TECH_GAP:-}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT_DIR"
export EXPORT_DIR OUT_DIR VIDEO_VIEWS TECH_GAP
echo "Export: $EXPORT_DIR"; echo "Video views: ${VIDEO_VIEWS:-none}"; echo "Results: $OUT_DIR"
if [ -n "$VIDEO_VIEWS" ] && [ -f "$VIDEO_VIEWS" ]; then echo "sha256 video_views: $(sha256sum "$VIDEO_VIEWS" | cut -c1-64)"; fi
if [ -f "$EXPORT_DIR/SHA256SUMS" ]; then (cd "$EXPORT_DIR" && sha256sum -c --quiet SHA256SUMS) && echo "SHA256SUMS: OK"; fi
run() { echo "== $1  $(date '+%H:%M:%S')"; "$PY" "$HERE/$1" > "$OUT_DIR/$2" 2>&1 || { echo "ERROR in $1, see $OUT_DIR/$2"; exit 1; }; }
run 10_step1_table1.py        step1_log.txt
run 11_step1_checks.py        step1_checks_log.txt
run 20_step2_tech_gap_poll.py step2_log.txt
run 30_pools.py               step3_pools_log.txt
run 31_step3_descriptive.py   step3_descriptive_log.txt
run 34_robustness.py          step3_robustness_log.txt
if [ -n "${COMMUNITY_ID:-}" ]; then export COMMUNITY_ID; run 33_community_profile.py step3_community_profile_log.txt; fi
run 32_step3_5_negbin.py      step3_5_log.txt          # slowest step (minutes)
run 35_share_cta.py           step3_cta_log.txt
run 40_key_numbers.py         key_numbers.txt
echo "Done: $OUT_DIR/key_numbers.txt"
