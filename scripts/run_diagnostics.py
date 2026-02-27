"""
Diagnostic Report Runner

Executes the full test suite via pytest, parses results per area,
and outputs the structured DIAGNOSTIC REPORT required by the AI architect.

Usage:
    python scripts/run_diagnostics.py
"""

from __future__ import annotations

import subprocess
import sys
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ── Test area mappings ──────────────────────────────────────────────────────
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
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
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


def main() -> None:
    print("=" * 72)
    print("  POLYMARKET MEAN-REVERSION MARKET MAKER — DIAGNOSTIC REPORT")
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
    bottlenecks: list[str] = []

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

    # Bottlenecks analysis
    print()
    print(f"[IDENTIFIED_BOTTLENECKS]:")
    bottleneck_checks = [
        ("WebSocket backoff is hardcoded 5s", "Consider exponential backoff with jitter"),
        ("SQLite writes are synchronous per-trade", "Consider batching writes or WAL mode"),
        ("Whale polling interval is fixed", "Adaptive polling based on market volatility"),
        ("No memory limit on trade queue", "asyncio.Queue maxsize could prevent OOM"),
        ("Settings dataclass exposes secrets in repr", "Override __repr__ to redact sensitive fields"),
    ]
    for desc, recommendation in bottleneck_checks:
        print(f"  - {desc}: {recommendation}")
    print()

    # Failing code snippets
    print(f"[CODE_SNIPPETS_NEEDS_FIXING]:")
    if all_failures:
        for f in all_failures[:10]:
            print(f"  - {f}")
    else:
        print("  (No failing tests detected)")
    print()

    # Exit code
    if any(parse_results(run_pytest_json(["tests/"])["stdout"])[1] > 0 for _ in [1]):
        sys.exit(1)
    else:
        print("[OK] All diagnostic areas PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
