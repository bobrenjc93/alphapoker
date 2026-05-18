#!/usr/bin/env bash
set -euo pipefail

uv run python -m alphapoker.train \
  --iterations "${ITERATIONS:-50000}" \
  --network-epochs "${NETWORK_EPOCHS:-0}" \
  --out "${OUT_DIR:-experiments/kuhn_cfr_baseline}"

