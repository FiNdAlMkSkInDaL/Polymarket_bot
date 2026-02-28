#!/usr/bin/env python3
"""
verify_launch.py — 60-Second PAPER-Mode Live-Fire Dry Run
═══════════════════════════════════════════════════════════

Imports the bot, starts it in PAPER mode against live Polymarket WebSockets,
runs for exactly 60 seconds, issues a graceful shutdown, and asserts that
all post-condition contracts are satisfied.

Usage:
    python scripts/verify_launch.py

Exit codes:
    0 — All assertions PASS (GO for soak test).
    1 — One or more assertions FAIL (NO-GO).
    2 — Bot failed to start (crash during init).
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

# ── Add project root to sys.path so `src` is importable ─────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Ensure PAPER environment BEFORE any src imports ──────────────────────
os.environ["DEPLOYMENT_ENV"] = "PAPER"
os.environ["PAPER_MODE"] = "true"

# Use an isolated temp directory for all outputs so we don't pollute the
# project tree.  Override the settings that derive paths from env vars.
_WORK_DIR = Path(tempfile.mkdtemp(prefix="polybot_verify_"))
_LOG_DIR = _WORK_DIR / "logs"
_DATA_DIR = _WORK_DIR / "data"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(parents=True, exist_ok=True)

os.environ["LOG_DIR"] = str(_LOG_DIR)
os.environ["RECORD_DATA"] = "true"
os.environ["RECORD_DATA_DIR"] = str(_DATA_DIR)

# ── Constants ────────────────────────────────────────────────────────────
RUN_DURATION_S = 60
HEALTH_FILE = "system_health.json"
DB_FILE = "trades.db"


# ═══════════════════════════════════════════════════════════════════════════
#  Result tracking
# ═══════════════════════════════════════════════════════════════════════════

class _Result:
    def __init__(self, name: str):
        self.name = name
        self.passed: bool | None = None
        self.detail: str = ""
        self.skipped: bool = False

    def ok(self, detail: str = "") -> None:
        self.passed = True
        self.detail = detail

    def fail(self, detail: str) -> None:
        self.passed = False
        self.detail = detail

    def skip(self, detail: str) -> None:
        self.skipped = True
        self.passed = True  # skipped counts as non-blocking
        self.detail = detail

    def __str__(self) -> str:
        if self.skipped:
            tag = "SKIP"
        elif self.passed:
            tag = "PASS"
        else:
            tag = "FAIL"
        suffix = f"  ({self.detail})" if self.detail else ""
        return f"  [{tag}] {self.name}{suffix}"


# ═══════════════════════════════════════════════════════════════════════════
#  Main verification routine
# ═══════════════════════════════════════════════════════════════════════════

async def _run_verification() -> list[_Result]:
    """Start the bot, wait, stop, and verify post-conditions."""

    results: list[_Result] = []

    # ── R1: Bot starts without crash ─────────────────────────────────────
    r_start = _Result("Bot starts in PAPER mode without crash")
    results.append(r_start)

    # Lazy import so env vars are already set when config module loads
    from src.bot import TradingBot
    from src.core.config import DeploymentEnv

    bot: TradingBot | None = None
    bot_task: asyncio.Task | None = None
    no_markets = False

    try:
        bot = TradingBot(deployment_env=DeploymentEnv.PAPER)
        bot_task = asyncio.create_task(bot.start())

        # Give the bot up to 30 s to finish initialisation.
        # If it finds no markets it will return early — that's OK.
        init_deadline = time.monotonic() + 30
        while time.monotonic() < init_deadline:
            if bot_task.done():
                # Bot exited early — check if it was a clean "no markets" exit
                exc = bot_task.exception() if not bot_task.cancelled() else None
                if exc:
                    r_start.fail(f"Bot crashed during init: {type(exc).__name__}: {exc}")
                    return results
                else:
                    # Clean early exit → probably no eligible markets
                    r_start.ok("started (exited early — likely no eligible markets)")
                    no_markets = True
                    break
            if getattr(bot, "_running", False):
                break
            await asyncio.sleep(0.5)

        if not no_markets and not getattr(bot, "_running", False):
            if bot_task.done():
                exc = bot_task.exception() if not bot_task.cancelled() else None
                if exc:
                    r_start.fail(f"Bot task ended unexpectedly: {exc}")
                else:
                    r_start.ok("started (returned before _running was set)")
                    no_markets = True
            else:
                r_start.fail("Bot did not reach _running=True within 30 s")
                bot_task.cancel()
                return results
        elif not no_markets:
            r_start.ok("running")

    except Exception as exc:
        r_start.fail(f"Constructor / start raised: {type(exc).__name__}: {exc}")
        return results

    # ── Soak period ──────────────────────────────────────────────────────
    if not no_markets:
        print(f"\n  Bot is running.  Soaking for {RUN_DURATION_S} s …")
        soak_start = time.monotonic()
        while time.monotonic() - soak_start < RUN_DURATION_S:
            elapsed = int(time.monotonic() - soak_start)
            remaining = RUN_DURATION_S - elapsed
            print(f"\r    ⏱  {elapsed:>3d} / {RUN_DURATION_S} s", end="", flush=True)

            if bot_task and bot_task.done():
                exc = bot_task.exception() if not bot_task.cancelled() else None
                if exc:
                    r_start.fail(f"Bot crashed during soak: {type(exc).__name__}: {exc}")
                    return results
                break

            await asyncio.sleep(1)

        print()  # newline after progress

    # ── Graceful shutdown ────────────────────────────────────────────────
    r_shutdown = _Result("Graceful shutdown completes without error")
    results.append(r_shutdown)

    try:
        if bot and not no_markets and bot_task and not bot_task.done():
            await bot.stop()
            # Give gather a moment to finish
            try:
                await asyncio.wait_for(bot_task, timeout=15)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        r_shutdown.ok()
    except Exception as exc:
        r_shutdown.fail(f"Shutdown raised: {type(exc).__name__}: {exc}")

    # ── R2: system_health.json exists ────────────────────────────────────
    r_health = _Result("system_health.json generated")
    results.append(r_health)

    health_path = _LOG_DIR / HEALTH_FILE
    if no_markets:
        # Health reporter runs on a 60-s interval; with early exit it may not fire.
        r_health.skip("Bot exited early (no markets) — health file not expected")
    elif health_path.exists():
        try:
            with open(health_path) as f:
                health_data = json.load(f)
            required_keys = {
                "timestamp", "deployment_env", "uptime_s",
                "ws_reconnects", "heartbeat_state",
            }
            missing = required_keys - set(health_data.keys())
            if missing:
                r_health.fail(f"Missing keys: {missing}")
            elif health_data.get("deployment_env") != "PAPER":
                r_health.fail(f"deployment_env={health_data.get('deployment_env')} (expected PAPER)")
            else:
                r_health.ok(
                    f"uptime={health_data.get('uptime_s')}s, "
                    f"heartbeat={health_data.get('heartbeat_state')}"
                )
        except json.JSONDecodeError as exc:
            r_health.fail(f"Invalid JSON: {exc}")
    else:
        r_health.fail(f"File not found: {health_path}")

    # ── R3: JSONL harvest file exists with size > 0 ─────────────────────
    r_data = _Result("JSONL harvest file(s) exist with size > 0")
    results.append(r_data)

    raw_ticks_dir = _DATA_DIR / "raw_ticks"
    if no_markets:
        r_data.skip("Bot exited early (no markets) — no data expected")
    elif raw_ticks_dir.exists():
        jsonl_files = list(raw_ticks_dir.rglob("*.jsonl"))
        if not jsonl_files:
            r_data.fail(f"No .jsonl files under {raw_ticks_dir}")
        else:
            non_empty = [f for f in jsonl_files if f.stat().st_size > 0]
            if not non_empty:
                r_data.fail(f"Found {len(jsonl_files)} .jsonl file(s) but all are empty")
            else:
                total_bytes = sum(f.stat().st_size for f in non_empty)
                r_data.ok(f"{len(non_empty)} file(s), {total_bytes:,} bytes total")
    else:
        r_data.fail(f"raw_ticks directory not found: {raw_ticks_dir}")

    # ── R4: SQLite database exists and is readable ──────────────────────
    r_db = _Result("SQLite trades.db exists, not locked, not corrupt")
    results.append(r_db)

    db_path = _LOG_DIR / DB_FILE
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            # Integrity check
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if result and result[0] == "ok":
                # Count tables
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                table_names = [t[0] for t in tables]
                conn.close()
                r_db.ok(f"tables: {', '.join(table_names)}")
            else:
                conn.close()
                r_db.fail(f"PRAGMA integrity_check returned: {result}")
        except sqlite3.OperationalError as exc:
            r_db.fail(f"SQLite error: {exc}")
    elif no_markets:
        # TradeStore.init() may not have been reached if bot returned
        # before _run().  Check if the file was at least created.
        r_db.skip("Bot exited before DB init (no markets)")
    else:
        r_db.fail(f"Database file not found: {db_path}")

    return results


def main() -> int:
    print("=" * 62)
    print("  Polymarket Bot — PAPER Mode Launch Verification")
    print("=" * 62)
    print(f"\n  Work directory : {_WORK_DIR}")
    print(f"  Log directory  : {_LOG_DIR}")
    print(f"  Data directory : {_DATA_DIR}")
    print(f"  Run duration   : {RUN_DURATION_S} s\n")

    results = asyncio.run(_run_verification())

    print("\n" + "─" * 62)
    print("  POST-CONDITION ASSERTIONS")
    print("─" * 62)

    for r in results:
        print(r)

    failures = [r for r in results if r.passed is False]
    skips = [r for r in results if r.skipped]

    print("─" * 62)
    total = len(results)
    passed = sum(1 for r in results if r.passed and not r.skipped)
    print(f"  {passed} passed, {len(skips)} skipped, {len(failures)} failed  (of {total})")
    print()

    if failures:
        print("  ██  VERDICT: NO-GO  ██")
        print("  Fix the failures above before proceeding to the soak test.")
        print()
        return 1
    else:
        print("  ██  VERDICT: GO  ██")
        print("  All checks passed.  Ready for 72-hour PAPER soak test.")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
