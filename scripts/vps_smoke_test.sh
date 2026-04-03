#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SWORD_SCAN_OUTPUT="$ROOT_DIR/config/live_executable_strips.json"
SWORD_SUMMARY_OUTPUT="$ROOT_DIR/data/clob_arb_launch_summary_paper.json"
SHIELD_REFRESH_OUTPUT="$ROOT_DIR/data/flb_results_live.json"
SHIELD_SUMMARY_OUTPUT="$ROOT_DIR/data/underwriter_launch_summary_paper.json"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

sanitize_linux_text_files() {
  find "$ROOT_DIR/scripts" \( -name '*.sh' -o -name '*.service' \) -print0 | while IFS= read -r -d '' file_path; do
    sed -i 's/\r$//' "$file_path"
  done
}

assert_json_file() {
  local path="$1"
  local expected_strategy="$2"
  if [[ ! -s "$path" ]]; then
    echo "Missing required JSON artifact: $path" >&2
    exit 1
  fi
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
path = Path(r'''$path''')
payload = json.loads(path.read_text(encoding='utf-8'))
if not isinstance(payload, dict):
    raise SystemExit(f'{path} is not a JSON object')
strategy = str(payload.get('strategy') or '')
if strategy not in {r'''$expected_strategy''', 'CLOB_GROUP_ARB'}:
    raise SystemExit(f'{path} strategy mismatch: {strategy!r}')
print(path.name)
PY
}

log "Sanitizing shell and service files for Linux"
sanitize_linux_text_files

rm -f "$SWORD_SUMMARY_OUTPUT" "$SHIELD_SUMMARY_OUTPUT"

log "Running one Sword PAPER cycle"
"$PYTHON_BIN" "$ROOT_DIR/scripts/live_bbo_arb_scanner.py" --output "$SWORD_SCAN_OUTPUT"
"$PYTHON_BIN" "$ROOT_DIR/scripts/launch_clob_arb.py" --env PAPER --input "$SWORD_SCAN_OUTPUT" --allow-empty --json-output "$SWORD_SUMMARY_OUTPUT"
assert_json_file "$SWORD_SUMMARY_OUTPUT" "SWORD"
"$PYTHON_BIN" "$ROOT_DIR/scripts/send_strategy_telegram_alert.py" sword --scan-summary "$SWORD_SCAN_OUTPUT" --launch-summary "$SWORD_SUMMARY_OUTPUT"

log "Running one Shield PAPER cycle"
"$PYTHON_BIN" "$ROOT_DIR/scripts/live_flb_scanner.py" --output "$SHIELD_REFRESH_OUTPUT"
"$PYTHON_BIN" "$ROOT_DIR/scripts/launch_underwriter.py" --env PAPER --input "$SHIELD_REFRESH_OUTPUT" --json-output "$SHIELD_SUMMARY_OUTPUT"
assert_json_file "$SHIELD_SUMMARY_OUTPUT" "SHIELD"
"$PYTHON_BIN" "$ROOT_DIR/scripts/send_strategy_telegram_alert.py" shield --launch-summary "$SHIELD_SUMMARY_OUTPUT"

log "Running strict Telegram comm check"
"$PYTHON_BIN" "$ROOT_DIR/scripts/send_strategy_telegram_alert.py" comm-check --message "VPS smoke test passed for Shield and Sword PAPER pipelines"

log "Smoke test passed; Sword and Shield PAPER artifacts generated successfully"