#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
PIPELINE="all"
RUN_ONCE=0
SWORD_INTERVAL_SECONDS="${SWORD_INTERVAL_SECONDS:-300}"
SHIELD_INTERVAL_SECONDS="${SHIELD_INTERVAL_SECONDS:-10800}"

SWORD_SCAN_OUTPUT="$ROOT_DIR/config/live_executable_strips.json"
SWORD_LAUNCH_SUMMARY="$ROOT_DIR/data/clob_arb_launch_summary_paper.json"
SHIELD_REFRESH_OUTPUT="$ROOT_DIR/data/flb_results_live.json"
SHIELD_LAUNCH_SUMMARY="$ROOT_DIR/data/underwriter_launch_summary_paper.json"

usage() {
  cat <<'EOF'
Usage: scripts/vps_master_scheduler.sh [--pipeline sword|shield|all] [--run-once]

Runs the autonomous PAPER pipelines for:
  sword  - live_bbo_arb_scanner.py -> launch_clob_arb.py --env PAPER
  shield - live_flb_scanner.py -> launch_underwriter.py --env PAPER

Environment overrides:
  PYTHON_BIN
  SWORD_INTERVAL_SECONDS
  SHIELD_INTERVAL_SECONDS
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

run_failure_alert() {
  local strategy="$1"
  local stage="$2"
  local message="$3"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/send_strategy_telegram_alert.py" failure --strategy "$strategy" --stage "$stage" --message "$message" || true
}

run_sword_once() {
  log "SWORD scan starting"
  if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/live_bbo_arb_scanner.py" --output "$SWORD_SCAN_OUTPUT"; then
    run_failure_alert "SWORD" "scanner" "live_bbo_arb_scanner.py failed"
    return 1
  fi

  local executable_strips
  executable_strips="$($PYTHON_BIN - <<PY
import json
from pathlib import Path
payload = json.loads(Path(r'''$SWORD_SCAN_OUTPUT''').read_text(encoding='utf-8'))
print(int(payload.get('executable_strips', 0) or 0))
PY
)"

  log "SWORD scan complete; executable strips=${executable_strips}"
  if [[ "$executable_strips" -le 0 ]]; then
    return 0
  fi

  if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/launch_clob_arb.py" --env PAPER --input "$SWORD_SCAN_OUTPUT" --json-output "$SWORD_LAUNCH_SUMMARY"; then
    run_failure_alert "SWORD" "launcher" "launch_clob_arb.py --env PAPER failed"
    return 1
  fi

  "$PYTHON_BIN" "$ROOT_DIR/scripts/send_strategy_telegram_alert.py" sword --scan-summary "$SWORD_SCAN_OUTPUT" --launch-summary "$SWORD_LAUNCH_SUMMARY" || true
  log "SWORD launch and alert complete"
}

run_shield_once() {
  log "SHIELD discovery starting"
  if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/live_flb_scanner.py" --output "$SHIELD_REFRESH_OUTPUT"; then
    run_failure_alert "SHIELD" "scanner" "live_flb_scanner.py failed"
    return 1
  fi

  local refreshed_active
  refreshed_active="$($PYTHON_BIN - <<PY
import json
from pathlib import Path
payload = json.loads(Path(r'''$SHIELD_REFRESH_OUTPUT''').read_text(encoding='utf-8'))
summary = payload.get('summary', {}) if isinstance(payload, dict) else {}
active_bucket = summary.get('active_bucket', {}) if isinstance(summary, dict) else {}
print(int(active_bucket.get('count', 0) or 0))
PY
)"

  log "SHIELD discovery complete; active targets=${refreshed_active}"
  if [[ "$refreshed_active" -le 0 ]]; then
    return 0
  fi

  if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/launch_underwriter.py" --env PAPER --input "$SHIELD_REFRESH_OUTPUT" --json-output "$SHIELD_LAUNCH_SUMMARY"; then
    run_failure_alert "SHIELD" "launcher" "launch_underwriter.py --env PAPER failed"
    return 1
  fi

  "$PYTHON_BIN" "$ROOT_DIR/scripts/send_strategy_telegram_alert.py" shield --launch-summary "$SHIELD_LAUNCH_SUMMARY" || true
  log "SHIELD launch and alert complete"
}

loop_sword() {
  while true; do
    run_sword_once || true
    [[ "$RUN_ONCE" -eq 1 ]] && return 0
    sleep "$SWORD_INTERVAL_SECONDS"
  done
}

loop_shield() {
  while true; do
    run_shield_once || true
    [[ "$RUN_ONCE" -eq 1 ]] && return 0
    sleep "$SHIELD_INTERVAL_SECONDS"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline)
      PIPELINE="$2"
      shift 2
      ;;
    --run-once)
      RUN_ONCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$PIPELINE" in
  sword)
    loop_sword
    ;;
  shield)
    loop_shield
    ;;
  all)
    loop_sword &
    sword_pid=$!
    loop_shield &
    shield_pid=$!
    wait -n "$sword_pid" "$shield_pid"
    exit_code=$?
    kill "$sword_pid" "$shield_pid" 2>/dev/null || true
    wait "$sword_pid" "$shield_pid" 2>/dev/null || true
    exit "$exit_code"
    ;;
  *)
    echo "Unsupported pipeline: $PIPELINE" >&2
    exit 2
    ;;
esac