#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# SCORCHED EARTH RESET — Total system reset for polymarket-bot
# Kills all processes, wipes all DBs/artefacts, pulls latest code,
# launches WFO pipeline + bot in fresh tmux sessions.
# Run on VPS as botuser.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

BOT_DIR="$HOME/polymarket-bot"
LOGS="$BOT_DIR/logs"

echo "════════════════════════════════════════════════════════════"
echo "  SCORCHED EARTH RESET — $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════"

# ── STEP 1: SHUTDOWN & PROCESS CLEANUP ─────────────────────────
echo ""
echo "[STEP 1] Killing all processes..."

echo "  → Killing bot processes..."
pkill -SIGTERM -f 'src.cli run' 2>/dev/null && echo "    killed src.cli" || echo "    no src.cli processes"
pkill -SIGTERM -f 'polybot run' 2>/dev/null && echo "    killed polybot" || echo "    no polybot processes"
sleep 2

echo "  → Killing WFO/optimization processes..."
pkill -SIGTERM -f 'run_optimization_pipeline' 2>/dev/null && echo "    killed optimizer" || echo "    no optimizer"
pkill -SIGTERM -f 'wfo_optimizer' 2>/dev/null && echo "    killed wfo" || echo "    no wfo"
pkill -SIGTERM -f 'optuna' 2>/dev/null && echo "    killed optuna" || echo "    no optuna"
sleep 2

echo "  → Cleaning shared memory segments..."
SHM_COUNT=$(ls /dev/shm/pmb_* 2>/dev/null | wc -l)
rm -f /dev/shm/pmb_*
echo "    removed $SHM_COUNT pmb_* segments"

echo "  → Killing tmux sessions..."
tmux kill-server 2>/dev/null && echo "    tmux server killed" || echo "    no tmux server running"

echo "  → Verifying no python processes remain..."
if pgrep -af python; then
    echo "    WARNING: python processes still running"
else
    echo "    ✓ no python processes"
fi

# Stop systemd service if it exists
sudo systemctl stop polymarket-bot.service 2>/dev/null && echo "  → systemd service stopped" || echo "  → no systemd service"
sudo systemctl disable polymarket-bot.service 2>/dev/null || true

# ── STEP 2: WIPE ALL PERSISTENT STATE ─────────────────────────
echo ""
echo "[STEP 2] Wiping persistent state & databases..."

echo "  → Removing trade database..."
rm -fv "$LOGS/trades.db" "$LOGS/trades.db-shm" "$LOGS/trades.db-wal"

echo "  → Removing Optuna WFO databases..."
rm -fv "$LOGS/wfo_phase1.db" "$LOGS/wfo_phase2.db" "$LOGS/wfo_phase3.db"

echo "  → Removing champion & bounds JSON artefacts..."
rm -fv "$LOGS"/*_champion.json "$LOGS"/*_bounds.json "$LOGS/final_validation_tearsheet.json"

echo "  → Removing system health snapshot..."
rm -fv "$LOGS/system_health.json"

echo "  → Clearing bot log files..."
> "$LOGS/bot.jsonl"
for f in "$LOGS"/bot.jsonl.*; do
    [ -f "$f" ] && rm -fv "$f"
done

echo "  → Clearing adverse selection history..."
> "$LOGS/adverse_sel_outcomes.jsonl" 2>/dev/null || true

echo "  → Verifying data/vps_march2026 is intact..."
if [ -d "$BOT_DIR/data/vps_march2026" ]; then
    PARQUET_COUNT=$(find "$BOT_DIR/data/vps_march2026" -name "*.parquet" | wc -l)
    echo "    ✓ data/vps_march2026 intact ($PARQUET_COUNT parquet files)"
else
    echo "    ✗ WARNING: data/vps_march2026 NOT FOUND!"
fi

# ── STEP 2.5: VERIFY CODE IS CURRENT ──────────────────────────
echo ""
echo "[STEP 2.5] Code version check..."
cd "$BOT_DIR"
echo "  current commit: $(git log --oneline -1)"

# ── STEP 3: VERIFY CONFIGURATION ──────────────────────────────
echo ""
echo "[STEP 3] Verifying configuration..."

echo "  → Checking config defaults..."
grep -n "NO_DISCOUNT_FACTOR.*0.995" src/core/config.py && echo "    ✓ NO_DISCOUNT_FACTOR = 0.995" || echo "    ✗ NO_DISCOUNT_FACTOR mismatch!"
grep -n "DRIFT_VOL_CEILING.*0.35" src/core/config.py && echo "    ✓ DRIFT_VOL_CEILING = 0.35" || echo "    ✗ DRIFT_VOL_CEILING mismatch!"
grep -n "ZSCORE_THRESHOLD.*0.20" src/core/config.py && echo "    ✓ ZSCORE_THRESHOLD = 0.20" || echo "    ✗ ZSCORE_THRESHOLD mismatch!"
grep -n "EQS_VOL_ADAPTIVE.*False" src/core/config.py && echo "    ✓ EQS_VOL_ADAPTIVE = False" || echo "    ✗ EQS_VOL_ADAPTIVE mismatch!"

echo "  → Checking iceberg_eqs_bonus clamp fix..."
if grep -q 'signal_q = min(1.0, signal_q + 0.15)' src/signals/edge_filter.py; then
    echo "    ✓ iceberg clamp fix in place"
else
    echo "    ✗ WARNING: iceberg clamp fix NOT found!"
fi

# ── STEP 4: VERIFY CLEAN SLATE ────────────────────────────────
echo ""
echo "[STEP 4] Verifying clean slate..."
echo "  trades.db:                $([ -f "$LOGS/trades.db" ] && echo 'EXISTS ✗' || echo 'GONE ✓')"
echo "  wfo_phase1.db:           $([ -f "$LOGS/wfo_phase1.db" ] && echo 'EXISTS ✗' || echo 'GONE ✓')"
echo "  wfo_phase2.db:           $([ -f "$LOGS/wfo_phase2.db" ] && echo 'EXISTS ✗' || echo 'GONE ✓')"
echo "  wfo_phase3.db:           $([ -f "$LOGS/wfo_phase3.db" ] && echo 'EXISTS ✗' || echo 'GONE ✓')"
echo "  champion JSONs:          $(ls "$LOGS"/*_champion.json 2>/dev/null | wc -l) remaining"
echo "  bounds JSONs:            $(ls "$LOGS"/*_bounds.json 2>/dev/null | wc -l) remaining"
echo "  validation tearsheet:    $([ -f "$LOGS/final_validation_tearsheet.json" ] && echo 'EXISTS ✗' || echo 'GONE ✓')"
echo "  system_health.json:      $([ -f "$LOGS/system_health.json" ] && echo 'EXISTS ✗' || echo 'GONE ✓')"
echo "  pmb_* shm segments:      $(ls /dev/shm/pmb_* 2>/dev/null | wc -l) remaining"

# ── STEP 5: LAUNCH FRESH SESSIONS ─────────────────────────────
echo ""
echo "[STEP 5] Launching fresh tmux sessions..."

cd "$BOT_DIR"

echo "  → Launching WFO pipeline in tmux 'wfo_fresh'..."
tmux new-session -d -s wfo_fresh \
  "cd $BOT_DIR && source .venv/bin/activate && python scripts/run_optimization_pipeline.py --data-dir data/vps_march2026 2>&1 | tee logs/wfo_fresh.log"
echo "    ✓ wfo_fresh session started"

sleep 2

echo "  → Launching bot in PENNY_LIVE mode in tmux 'bot_fresh'..."
tmux new-session -d -s bot_fresh \
  "cd $BOT_DIR && source .venv/bin/activate && python -m src.cli run --env PENNY_LIVE 2>&1 | tee logs/bot_fresh.log"
echo "    ✓ bot_fresh session started"

sleep 5

# ── STEP 6: FINAL CONFIRMATION ────────────────────────────────
echo ""
echo "[STEP 6] Final confirmation..."

echo "  → tmux sessions:"
tmux ls 2>&1

echo "  → trades.db initialized: $([ -f "$LOGS/trades.db" ] && echo 'YES ✓' || echo 'NOT YET (may take a moment)')"

echo "  → Recent bot log output:"
sleep 5
tail -20 "$LOGS/bot.jsonl" 2>/dev/null || echo "    (no output yet)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  SCORCHED EARTH RESET COMPLETE — $(date -Iseconds)"
echo "  Monitor with: tmux attach -t bot_fresh"
echo "  WFO progress: tmux attach -t wfo_fresh"
echo "════════════════════════════════════════════════════════════"
