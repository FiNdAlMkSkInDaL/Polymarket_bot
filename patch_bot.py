#!/usr/bin/env python3
"""Patch bot.py to fix two root-cause bugs preventing trades."""

import sys

BOT_PY = "/home/botuser/polymarket-bot/src/bot.py"

with open(BOT_PY, "r") as f:
    content = f.read()

lines = content.split("\n")

# ──────────────────────────────────────────────────────────────────
# FIX 1: _regime_score UnboundLocalError
#
# The variable _regime_score is assigned INSIDE the `if sig:` block
# but referenced in the `if not sig:` (drift signal) block.
# When panic doesn't fire but drift does with BUY_NO direction,
# _regime_score is undefined → UnboundLocalError crashes the
# _stale_bar_flush_loop silently.
#
# Fix: Move the assignment BEFORE `if sig:` so both branches see it.
# ──────────────────────────────────────────────────────────────────

target_sig_line = None
for i, line in enumerate(lines):
    if "sig = detector.evaluate(bar, no_best_ask=no_best_ask, whale_confluence=whale)" in line:
        target_sig_line = i
        break

if target_sig_line is None:
    print("ERROR: Could not find sig = detector.evaluate line")
    sys.exit(1)

# Verify structure
if "if sig:" not in lines[target_sig_line + 1]:
    print(f"ERROR: Expected 'if sig:' at line {target_sig_line + 2}, got: {lines[target_sig_line + 1]!r}")
    sys.exit(1)

# Find  _regime_score inside the if sig: block (should be 2-3 lines below)
regime_offset = None
for offset in range(2, 6):
    if "_regime_score = regime_det.regime_score" in lines[target_sig_line + offset]:
        regime_offset = offset
        break

if regime_offset is None:
    print("ERROR: Could not find _regime_score assignment inside if sig: block")
    sys.exit(1)

print(f"Found sig= at line {target_sig_line + 1}, _regime_score at line {target_sig_line + regime_offset + 1}")

# Insert _regime_score assignment AFTER sig= line, at same indentation as sig=
new_regime_line = "            _regime_score = regime_det.regime_score if regime_det else 0.5"
lines.insert(target_sig_line + 1, new_regime_line)

# Remove the old _regime_score line (shifted by 1 due to insert)
old_idx = target_sig_line + regime_offset + 1
del lines[old_idx]

print("FIX 1 applied: _regime_score moved before if sig: block")

# ──────────────────────────────────────────────────────────────────
# FIX 2: Heartbeat position_book_stale threshold too tight
#
# The position-level stale check uses self._stale_ms * 3 = 15000ms.
# Prediction market L2 snapshots arrive every ~15s on low-volume
# markets, triggering false stale detection.  The result is that
# every paper order gets cancelled within ~25s of placement.
#
# Fix: Change the multiplier from 3 to 6 (= 30s threshold),
# which accommodates the natural update cadence.
# ──────────────────────────────────────────────────────────────────

heartbeat_fixed = False
for i, line in enumerate(lines):
    if "position_stale_ms = self._stale_ms * 3" in line:
        lines[i] = line.replace("self._stale_ms * 3", "self._stale_ms * 6")
        heartbeat_fixed = True
        print(f"FIX 2 applied: position stale multiplier 3 → 6 at line {i + 1}")
        break

if not heartbeat_fixed:
    print("WARNING: Could not find position_stale_ms multiplier line")

# Write back
with open(BOT_PY, "w") as f:
    f.write("\n".join(lines))

print("All fixes saved to", BOT_PY)
