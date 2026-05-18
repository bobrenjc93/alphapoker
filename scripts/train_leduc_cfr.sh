#!/usr/bin/env bash
set -euo pipefail

uv run python -m alphapoker.train_leduc \
  --iterations "${ITERATIONS:-100}" \
  --out "${OUT_DIR:-experiments/leduc_cfr_smoke}"

