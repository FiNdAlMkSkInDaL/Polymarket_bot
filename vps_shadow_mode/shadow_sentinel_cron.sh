#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/botuser/polymarket-bot"
SHADOW_ROOT="$REPO_ROOT/shadow_mode"
LOG_ROOT="$REPO_ROOT/shadow_logs"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

RUN_DATE="$(date -u +%F)"
RUN_STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
RUN_ROOT="$LOG_ROOT/$RUN_STAMP"
mkdir -p "$RUN_ROOT"

cd "$SHADOW_ROOT"

"$PYTHON_BIN" scripts/refresh_shadow_metadata_cache.py \
  --lake-root "$REPO_ROOT/data/l2_book_live" \
  --cache-path "$SHADOW_ROOT/artifacts/clob_arb_baseline_metadata.json" \
  --summary-path "$RUN_ROOT/metadata_refresh_preflight.json"

"$PYTHON_BIN" scripts/run_scavenger_protocol_historical_sweep.py \
  --output-root "$RUN_ROOT/scavenger_protocol_historical_sweep"

"$PYTHON_BIN" scripts/run_conditional_probability_squeeze_batch.py \
  --output-dir "$RUN_ROOT/conditional_probability_squeeze_batch" \
  --start-date "$RUN_DATE" \
  --end-date "$RUN_DATE"

"$PYTHON_BIN" scripts/run_mid_tier_probability_compression_historical_sweep.py \
  --output-dir "$RUN_ROOT/mid_tier_probability_compression_historical_sweep" \
  --start-date "$RUN_DATE" \
  --end-date "$RUN_DATE"