#!/usr/bin/env bash
set -uo pipefail

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

PIPELINE_EXIT_CODE=0

run_step() {
  local step_name="$1"
  shift
  if "$@"; then
    echo "SHADOW_STEP_OK step=$step_name"
    return 0
  fi
  local step_exit=$?
  PIPELINE_EXIT_CODE=1
  echo "WARNING SHADOW_STEP_FAILED step=$step_name exit_code=$step_exit"
  return 0
}

run_step metadata_preflight \
  "$PYTHON_BIN" scripts/refresh_shadow_metadata_cache.py \
  --lake-root "$REPO_ROOT/data/l2_book_live" \
  --cache-path "$SHADOW_ROOT/artifacts/clob_arb_baseline_metadata.json" \
  --summary-path "$RUN_ROOT/metadata_refresh_preflight.json"

run_step scavenger \
  "$PYTHON_BIN" scripts/run_scavenger_protocol_historical_sweep.py \
  --output-root "$RUN_ROOT/scavenger_protocol_historical_sweep"

run_step squeeze \
  "$PYTHON_BIN" scripts/run_conditional_probability_squeeze_batch.py \
  --output-dir "$RUN_ROOT/conditional_probability_squeeze_batch" \
  --start-date "$RUN_DATE" \
  --end-date "$RUN_DATE"

run_step mid_tier \
  "$PYTHON_BIN" scripts/run_mid_tier_probability_compression_historical_sweep.py \
  --output-dir "$RUN_ROOT/mid_tier_probability_compression_historical_sweep" \
  --start-date "$RUN_DATE" \
  --end-date "$RUN_DATE"

"$PYTHON_BIN" scripts/send_shadow_hourly_telegram.py \
  --run-root "$RUN_ROOT" || true

exit "$PIPELINE_EXIT_CODE"