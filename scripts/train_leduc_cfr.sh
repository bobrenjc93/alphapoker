#!/usr/bin/env bash
set -euo pipefail

args=(
  python -m alphapoker.train_leduc
  --iterations "${ITERATIONS:-100}"
  --out "${OUT_DIR:-experiments/leduc_cfr_smoke}"
)

if [[ "${BEST_RESPONSE:-0}" == "1" ]]; then
  args+=(--best-response)
fi
if [[ -n "${CHECKPOINT_IN:-}" ]]; then
  args+=(--checkpoint-in "$CHECKPOINT_IN")
fi
if [[ -n "${CHECKPOINT_OUT:-}" ]]; then
  args+=(--checkpoint-out "$CHECKPOINT_OUT")
fi

uv run "${args[@]}"
