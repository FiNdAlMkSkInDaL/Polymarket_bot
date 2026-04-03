from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify VPS warm-ready state for hybrid cutover.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/botuser/polymarket-bot"),
        help="Project root on the VPS.",
    )
    parser.add_argument(
        "--python-path",
        type=Path,
        default=Path("/home/botuser/polymarket-bot/.venv/bin/python3"),
        help="Python executable used by the staged ExecStart.",
    )
    return parser.parse_args()


def _green(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="GREEN", detail=detail)


def _red(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="RED", detail=detail)


def _check_file_integrity(root: Path) -> CheckResult:
    required_files = [
        root / "scripts" / "launch_hybrid_arb.py",
        root / "src" / "data" / "alchemy_rpc_client.py",
        root / "src" / "signals" / "hybrid_arb_maker.py",
    ]
    missing_or_empty: list[str] = []
    present: list[str] = []
    for path in required_files:
        if not path.exists():
            missing_or_empty.append(f"missing:{path}")
            continue
        size = path.stat().st_size
        if size <= 0:
            missing_or_empty.append(f"empty:{path}")
            continue
        present.append(f"{path.name}={size}B")
    if missing_or_empty:
        return _red("File Integrity", "; ".join(missing_or_empty))
    return _green("File Integrity", ", ".join(present))


def _check_env(root: Path) -> CheckResult:
    env_path = root / ".env"
    if not env_path.exists():
        return _red("Env Validation", f"missing:{env_path}")
    value = ""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ALCHEMY_POLYGON_RPC_URL="):
            value = line.split("=", 1)[1].strip()
            break
    if not value:
        return _red("Env Validation", "ALCHEMY_POLYGON_RPC_URL missing or empty")
    return _green("Env Validation", f"ALCHEMY_POLYGON_RPC_URL present len={len(value)}")


def _check_venv(python_path: Path, root: Path) -> CheckResult:
    if not python_path.exists():
        return _red("Venv Check", f"missing:{python_path}")
    try:
        completed = subprocess.run(
            [
                str(python_path),
                "-c",
                "import requests, web3; print('ok')",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return _red("Venv Check", f"execution_failed:{type(exc).__name__}: {exc}")
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or f"exit={completed.returncode}"
        return _red("Venv Check", stderr)
    return _green("Venv Check", f"python={python_path}")


def _dir_is_writable(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing:{path}"
    if not path.is_dir():
        return False, f"not_dir:{path}"
    if not os.access(path, os.W_OK):
        return False, f"not_writable:{path}"
    probe = path / ".vps_health_check_probe"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return False, f"probe_failed:{path}:{type(exc).__name__}: {exc}"
    return True, f"writable:{path}"


def _check_permissions(root: Path) -> CheckResult:
    targets = [root / "logs", root / "data"]
    failures: list[str] = []
    successes: list[str] = []
    for target in targets:
        ok, detail = _dir_is_writable(target)
        if ok:
            successes.append(detail)
        else:
            failures.append(detail)
    if failures:
        return _red("Permission Audit", "; ".join(failures))
    return _green("Permission Audit", "; ".join(successes))


def main() -> int:
    args = _parse_args()
    results = [
        _check_file_integrity(args.root),
        _check_env(args.root),
        _check_venv(args.python_path, args.root),
        _check_permissions(args.root),
    ]
    overall_green = all(result.status == "GREEN" for result in results)
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")
    print(f"[{'GREEN' if overall_green else 'RED'}] Overall: {'Warm-Ready' if overall_green else 'Not Ready'}")
    return 0 if overall_green else 1


if __name__ == "__main__":
    raise SystemExit(main())