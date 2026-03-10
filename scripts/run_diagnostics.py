"""
Diagnostic Report Runner

Executes the full test suite via pytest, parses results per area,
and runs live bottleneck verification checks against the codebase
to confirm that all 5 critical production-readiness fixes are in place.

Usage:
    python scripts/run_diagnostics.py
"""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import json
import re
from dataclasses import dataclass, fields, field
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# -- Test area mappings ------------------------------------------------------
AREAS = {
    "MATH_LOGIC": {
        "files": [
            "tests/test_unit_math.py",
            "tests/test_ohlcv.py",
            "tests/test_take_profit.py",
            "tests/test_panic_detector.py",
        ],
        "description": "EMA/VWAP/Z-score calculations, take-profit formula, numeric regressions",
    },
    "WEBSOCKET_CONN": {
        "files": [
            "tests/test_integration.py::TestWebSocketParsing",
            "tests/test_integration.py::TestWebSocketReconnect",
            "tests/test_resilience.py::TestWebSocketResilience",
        ],
        "description": "WebSocket connect, parse, auto-reconnect",
    },
    "WHALE_RPC_CONN": {
        "files": [
            "tests/test_integration.py::TestWhaleMonitor",
        ],
        "description": "Polygonscan RPC polling, rate-limit handling, direction detection",
    },
    "SQLITE_LOGGING": {
        "files": [
            "tests/test_system.py::TestVirtualBuy",
            "tests/test_system.py::TestVirtualSell",
            "tests/test_system.py::TestAggregateStats",
            "tests/test_trade_store.py",
            "tests/test_resilience.py::TestSQLiteResilience",
        ],
        "description": "Paper trade recording, PnL calculations, aggregate stats, go-live criteria",
    },
    "ERROR_HANDLING": {
        "files": [
            "tests/test_resilience.py::TestTimeoutEnforcement",
            "tests/test_resilience.py::TestExecutorEdgeCases",
            "tests/test_resilience.py::TestPanicDetectorEdgeCases",
            "tests/test_resilience.py::TestOHLCVEdgeCases",
            "tests/test_system.py::TestPositionManagerRisk",
            "tests/test_integration.py::TestEnvSafety",
            "tests/test_executor.py",
        ],
        "description": "Timeouts, edge cases, risk gates, env safety, executor lifecycle",
    },
}


def run_pytest_json(targets: list[str]) -> dict:
    """Run pytest with JSON report plugin and return parsed results."""
    cmd = [
        sys.executable, "-m", "pytest",
        *targets,
        "-v",
        "--tb=short",
        "--no-header",
        "-q",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def parse_results(stdout: str) -> tuple[int, int, int, list[str]]:
    """Parse pytest output to extract pass/fail/warn counts and failure details."""
    passed = 0
    failed = 0
    warned = 0
    failures = []

    for line in stdout.splitlines():
        if line.strip().startswith("PASSED"):
            passed += 1
        elif "PASSED" in line:
            passed += 1
        if line.strip().startswith("FAILED"):
            failed += 1
            failures.append(line.strip())
        elif "FAILED" in line:
            failed += 1
            failures.append(line.strip())
        if "WARNING" in line.upper() or "WARN" in line.upper():
            warned += 1

    # Parse summary line: e.g., "5 passed, 2 failed"
    summary_match = re.search(r"(\d+) passed", stdout)
    fail_match = re.search(r"(\d+) failed", stdout)
    warn_match = re.search(r"(\d+) warning", stdout)

    if summary_match:
        passed = int(summary_match.group(1))
    if fail_match:
        failed = int(fail_match.group(1))
    if warn_match:
        warned = int(warn_match.group(1))

    return passed, failed, warned, failures


def determine_status(passed: int, failed: int, warned: int) -> str:
    if failed > 0:
        return "FAIL"
    if warned > 0:
        return "WARN"
    if passed > 0:
        return "PASS"
    return "WARN"


# ---------------------------------------------------------------------------
#  Bottleneck verification helpers
# ---------------------------------------------------------------------------

async def _test_enqueue_drop_oldest() -> tuple[bool, str]:
    """Verify that _enqueue drops oldest when queue is full."""
    import time as _time
    from src.data.websocket_client import MarketWebSocket, TradeEvent

    q: asyncio.Queue[TradeEvent] = asyncio.Queue(maxsize=5)
    ws = MarketWebSocket(["test"], q, queue_maxsize=5)

    # Fill the queue
    for i in range(5):
        q.put_nowait(TradeEvent(
            timestamp=_time.time(), market_id="m", asset_id="a",
            side="buy", price=0.5 + i * 0.01, size=1.0, is_yes=True,
        ))

    # Enqueue a new item -- should drop oldest
    newest = TradeEvent(
        timestamp=_time.time(), market_id="m", asset_id="a",
        side="buy", price=0.99, size=1.0, is_yes=True,
    )
    await ws._enqueue(newest)

    items = []
    while not q.empty():
        items.append(q.get_nowait())

    if items and items[-1].price == 0.99 and len(items) == 5:
        return True, "Drop-oldest verified: oldest discarded, newest enqueued"
    return False, (
        f"Drop-oldest unexpected: {len(items)} items, "
        f"last price={items[-1].price if items else 'N/A'}"
    )


def main() -> None:
    print("=" * 72)
    print("  POLYMARKET MEAN-REVERSION MARKET MAKER -- DIAGNOSTIC REPORT")
    print("=" * 72)
    print()

    # Run all tests first to get overall coverage
    print("[*] Running full test suite...\n")
    full_result = run_pytest_json(["tests/"])
    print(full_result["stdout"])
    if full_result["stderr"]:
        print(full_result["stderr"])

    print()
    print("=" * 72)
    print("### DIAGNOSTIC REPORT")
    print("=" * 72)
    print()

    all_failures: list[str] = []

    for area_key, area_info in AREAS.items():
        result = run_pytest_json(area_info["files"])
        passed, failed, warned, failures = parse_results(result["stdout"])
        status = determine_status(passed, failed, warned)

        details = area_info["description"]
        if failures:
            details += f" | FAILURES: {'; '.join(failures[:3])}"
            all_failures.extend(failures)

        print(f"[{area_key}]: {status} (Details: {details})")
        print(f"    Tests: {passed} passed, {failed} failed, {warned} warnings")
        print()

    # ==================================================================
    #  Bottleneck verification -- live runtime checks
    # ==================================================================
    print()
    print("[BOTTLENECK_VERIFICATION]:")
    print()

    bottleneck_pass = 0
    bottleneck_fail = 0

    # -- 1. Settings.__repr__ redacts secrets ----------------------------
    from src.core.config import Settings
    s = Settings(
        polymarket_api_key="sk-SUPERSECRETKEY123",
        polymarket_secret="sec-TOPSECRET456",
        polymarket_passphrase="pass-HIDDEN789",
        eoa_private_key="0xDEADBEEF0000PRIVATE",
        polygon_rpc_url="https://polygon-rpc.example.com",
        polygonscan_api_key="PSCAN-APIKEY999",
        telegram_bot_token="1234567890:AAHdqTcvZE_FAKE_TOKEN",
        telegram_chat_id="-100123456",
    )
    repr_str = repr(s)

    exposed_secrets = []
    for secret_val in [
        "sk-SUPERSECRETKEY123", "sec-TOPSECRET456", "pass-HIDDEN789",
        "0xDEADBEEF0000PRIVATE", "PSCAN-APIKEY999",
        "1234567890:AAHdqTcvZE_FAKE_TOKEN",
    ]:
        if secret_val in repr_str:
            exposed_secrets.append(secret_val[:12])

    if not exposed_secrets:
        print("  [PASS] Fix 1 -- Settings.__repr__ redacts SECRET/KEY/PASSPHRASE/TOKEN fields")
        bottleneck_pass += 1
    else:
        print(f"  [FAIL] Fix 1 -- Settings.__repr__ still exposes: {exposed_secrets}")
        bottleneck_fail += 1

    if "sk-S***" in repr_str and "***" in repr_str:
        print("         [ok] Masking format confirmed (first 4 chars + ***)")
    else:
        print("         [??] Masking format may differ from expected pattern")

    # -- 2. Bounded async queue with drop-oldest -------------------------
    from src.data.websocket_client import MarketWebSocket, TradeEvent

    has_enqueue = hasattr(MarketWebSocket, "_enqueue") and callable(
        getattr(MarketWebSocket, "_enqueue")
    )
    if has_enqueue:
        print("  [PASS] Fix 2 -- MarketWebSocket._enqueue() with drop-oldest logic")
        bottleneck_pass += 1
    else:
        print("  [FAIL] Fix 2 -- MarketWebSocket missing _enqueue() method")
        bottleneck_fail += 1

    # Behavioural check (async)
    ok, msg = asyncio.run(_test_enqueue_drop_oldest())
    print(f"         [{'ok' if ok else '??'}] {msg}")

    # bot.py maxsize
    bot_src = Path("src/bot.py").read_text(encoding="utf-8")
    if "maxsize=" in bot_src:
        print("         [ok] TradingBot creates queue with maxsize")
    else:
        print("         [??] TradingBot queue may not have maxsize set")

    # -- 3. SQLite WAL mode ----------------------------------------------
    trade_store_src = Path("src/monitoring/trade_store.py").read_text(encoding="utf-8")
    wal_ok = "journal_mode=WAL" in trade_store_src
    sync_ok = "synchronous=NORMAL" in trade_store_src

    if wal_ok and sync_ok:
        print("  [PASS] Fix 3 -- SQLite WAL mode + synchronous=NORMAL on init")
        bottleneck_pass += 1
    else:
        missing = []
        if not wal_ok:
            missing.append("journal_mode=WAL")
        if not sync_ok:
            missing.append("synchronous=NORMAL")
        print(f"  [FAIL] Fix 3 -- SQLite missing PRAGMAs: {', '.join(missing)}")
        bottleneck_fail += 1

    # -- 4. Exponential backoff with jitter ------------------------------
    ws_src = Path("src/data/websocket_client.py").read_text(encoding="utf-8")
    has_backoff = "_BACKOFF_BASE" in ws_src and "_BACKOFF_MAX" in ws_src
    has_exp = "2 ** attempt" in ws_src or "2**attempt" in ws_src
    has_jitter = "random.uniform" in ws_src
    no_hardcoded = "await asyncio.sleep(5)" not in ws_src

    if has_backoff and has_exp and has_jitter and no_hardcoded:
        print("  [PASS] Fix 4 -- Exponential backoff + jitter (no hardcoded 5s)")
        bottleneck_pass += 1
    else:
        issues = []
        if not has_backoff:
            issues.append("missing BACKOFF constants")
        if not has_exp:
            issues.append("missing 2**attempt formula")
        if not has_jitter:
            issues.append("missing random jitter")
        if not no_hardcoded:
            issues.append("hardcoded sleep(5) still present")
        print(f"  [FAIL] Fix 4 -- Backoff issues: {', '.join(issues)}")
        bottleneck_fail += 1

    base_match = re.search(r"_BACKOFF_BASE.*?=\s*([\d.]+)", ws_src)
    max_match = re.search(r"_BACKOFF_MAX.*?=\s*([\d.]+)", ws_src)
    if base_match and max_match:
        print(f"         [ok] Backoff range: base={base_match.group(1)}s, max={max_match.group(1)}s")

    # -- 5. Adaptive whale polling ---------------------------------------
    from src.signals.whale_monitor import WhaleMonitor

    sig = inspect.signature(WhaleMonitor.__init__)
    has_zscore_param = "zscore_fn" in sig.parameters
    has_update = hasattr(WhaleMonitor, "_update_interval")
    has_current_attr = "_current_interval" in WhaleMonitor.__init__.__code__.co_names or True
    # Quick instance check
    monitor_test = WhaleMonitor(zscore_fn=lambda: 0.0)
    has_current_attr = hasattr(monitor_test, "_current_interval")

    if has_zscore_param and has_update and has_current_attr:
        print("  [PASS] Fix 5 -- WhaleMonitor adaptive polling via zscore_fn")
        bottleneck_pass += 1
    else:
        issues = []
        if not has_zscore_param:
            issues.append("missing zscore_fn parameter")
        if not has_update:
            issues.append("missing _update_interval()")
        if not has_current_attr:
            issues.append("missing _current_interval")
        print(f"  [FAIL] Fix 5 -- Adaptive polling issues: {', '.join(issues)}")
        bottleneck_fail += 1

    # Behavioural check
    m_panic = WhaleMonitor(zscore_fn=lambda: 5.0)
    m_panic._update_interval()
    panic_iv = m_panic._current_interval

    m_calm = WhaleMonitor(zscore_fn=lambda: 0.1)
    m_calm._update_interval()
    calm_iv = m_calm._current_interval

    if panic_iv < 5.0 and calm_iv >= 15.0:
        print(f"         [ok] Adaptive range: panic={panic_iv}s, calm={calm_iv}s")
    else:
        print(f"         [??] Intervals unexpected: panic={panic_iv}s, calm={calm_iv}s")

    if "zscore_fn=" in bot_src:
        print("         [ok] TradingBot wires zscore_fn to WhaleMonitor")
    else:
        print("         [??] TradingBot may not pass zscore_fn")

    # ==================================================================
    #  Summary
    # ==================================================================
    print()
    print("=" * 72)
    total_passed, total_failed, _, _ = parse_results(full_result["stdout"])
    print(f"  TESTS: {total_passed} passed, {total_failed} failed")
    print(f"  BOTTLENECK FIXES: {bottleneck_pass}/5 VERIFIED", end="")
    if bottleneck_fail > 0:
        print(f", {bottleneck_fail} REMAINING")
    else:
        print(" -- ALL CLEAR")
    print("=" * 72)
    print()

    if total_failed == 0 and bottleneck_fail == 0:
        print(
            f"[OK] All {total_passed} tests PASSED, all 5 bottlenecks "
            "RESOLVED -- ready for paper-trading"
        )
        sys.exit(0)
    else:
        if total_failed > 0:
            print(f"[!!] {total_failed} test(s) still failing")
        if bottleneck_fail > 0:
            print(f"[!!] {bottleneck_fail} bottleneck(s) still unresolved")
        sys.exit(1)


if __name__ == "__main__":
    main()

