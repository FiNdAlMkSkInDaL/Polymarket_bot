#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable

os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.enrich_lake_metadata import DEFAULT_BATCH_SIZE, DEFAULT_TIMEOUT_SECONDS, GammaMarketRow, run_enrichment


DEFAULT_REMOTE = "botuser@135.181.85.32"
DEFAULT_REMOTE_ROOT = "/home/botuser/polymarket-bot/data/l2_book_live"
DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "artifacts" / "l2_parquet_lake_rolling" / "l2_book"
DEFAULT_INTERVAL_SECONDS = 3600.0
DEPTH_LEVELS = 5
MANIFEST_NAME = "manifest.json"
ENRICHED_MANIFEST_NAME = "enriched_manifest.json"
SYNC_STATE_NAME = "sync_state.json"
HANDOFF_FILE_NAME = "writer_handoff.json"
HANDOFF_RELATIVE_PATH = Path("_state") / HANDOFF_FILE_NAME

STRICT_SCHEMA = {
    "timestamp": "Datetime(ms, UTC)",
    "market_id": "Utf8",
    "event_id": "Utf8",
    "token_id": "Utf8",
    "best_bid": "Float64",
    "best_ask": "Float64",
    "bid_depth": "Float64",
    "ask_depth": "Float64",
}

SCAN_SCHEMA = {
    **STRICT_SCHEMA,
    "date": "Utf8 partition YYYY-MM-DD",
    "hour": "Utf8 partition HH",
}


@dataclass(frozen=True, slots=True)
class SyncTool:
    name: str
    command: list[str]
    transfer_family: str = "scp"
    path_style: str = "windows"


@dataclass(frozen=True, slots=True)
class RemoteFileSnapshot:
    relative_path: PurePosixPath
    size: int
    mtime_ns: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync live clean-lake parquet shards from the Helsinki VPS into a rolling local lake.",
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="SSH destination, e.g. botuser@135.181.85.32.")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Remote clean-lake root directory.")
    parser.add_argument(
        "--local-root",
        type=Path,
        default=DEFAULT_LOCAL_ROOT,
        help="Local parquet destination root. If the name is l2_book, manifest files are written to the parent.",
    )
    parser.add_argument(
        "--subpath",
        default="",
        help="Optional partition subpath such as date=2026-04-04/hour=18.",
    )
    parser.add_argument("--tool", choices=("auto", "rsync", "scp"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Preview rsync operations without copying files.")
    parser.add_argument("--delete", action="store_true", help="Mirror deletions locally when using rsync.")
    parser.add_argument("--loop", action="store_true", help="Continue syncing forever instead of exiting after one iteration.")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Sleep interval between iterations when --loop is set.",
    )
    parser.add_argument(
        "--min-date",
        default="",
        help="Optional YYYY-MM-DD partition floor to keep locally after sync.",
    )
    parser.add_argument(
        "--gamma-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of market ids to request per Gamma batch when refreshing enriched_manifest.json.",
    )
    parser.add_argument(
        "--gamma-timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout for Gamma enrichment refreshes.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _clean_subpath(raw_value: str) -> PurePosixPath | None:
    text = str(raw_value or "").strip().replace("\\", "/").strip("/")
    if not text:
        return None
    subpath = PurePosixPath(text)
    if subpath.is_absolute() or ".." in subpath.parts:
        raise ValueError(f"Invalid subpath: {raw_value!r}")
    return subpath


def _normalize_min_date(raw_value: str) -> str | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid --min-date value: {raw_value!r}") from exc
    return text


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part for part in resolved.parts[1:] if part not in ("/", "\\")]
    return "/mnt/" + "/".join([drive, *parts])


def _to_msys_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part for part in resolved.parts[1:] if part not in ("/", "\\")]
    return "/" + "/".join([drive, *parts])


def _sync_tool_for_rsync_path(path: str) -> SyncTool:
    normalized = path.replace("/", "\\").lower()
    if "\\git\\usr\\bin\\rsync.exe" in normalized:
        return SyncTool(name="git-rsync", command=[path], transfer_family="rsync", path_style="msys")
    if "\\msys64\\" in normalized and normalized.endswith("\\rsync.exe"):
        return SyncTool(name="msys-rsync", command=[path], transfer_family="rsync", path_style="msys")
    return SyncTool(name="rsync", command=[path], transfer_family="rsync")


def _find_rsync_tool() -> SyncTool | None:
    rsync_path = shutil.which("rsync")
    if rsync_path:
        return _sync_tool_for_rsync_path(rsync_path)

    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_name)
        if not base:
            continue
        candidate = Path(base) / "Git" / "usr" / "bin" / "rsync.exe"
        if candidate.exists():
            return _sync_tool_for_rsync_path(str(candidate))

    for candidate in (
        Path("C:/msys64/usr/bin/rsync.exe"),
        Path("C:/msys64/ucrt64/bin/rsync.exe"),
    ):
        if candidate.exists():
            return _sync_tool_for_rsync_path(str(candidate))

    wsl_path = shutil.which("wsl")
    if wsl_path and _wsl_rsync_available(wsl_path):
        return SyncTool(name="wsl-rsync", command=[wsl_path, "rsync"], transfer_family="rsync", path_style="wsl")
    return None


def _wsl_rsync_available(wsl_path: str) -> bool:
    try:
        completed = subprocess.run(
            [wsl_path, "-e", "sh", "-lc", "command -v rsync >/dev/null 2>&1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _find_scp_tool() -> SyncTool | None:
    scp_path = shutil.which("scp")
    if scp_path:
        return SyncTool(name="scp", command=[scp_path], transfer_family="scp")
    candidate = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32" / "OpenSSH" / "scp.exe"
    if candidate.exists():
        return SyncTool(name="scp", command=[str(candidate)], transfer_family="scp")
    return None


def _find_ssh_tool() -> SyncTool | None:
    ssh_path = shutil.which("ssh")
    if ssh_path:
        return SyncTool(name="ssh", command=[ssh_path], transfer_family="ssh")
    candidate = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32" / "OpenSSH" / "ssh.exe"
    if candidate.exists():
        return SyncTool(name="ssh", command=[str(candidate)], transfer_family="ssh")
    return None


def _choose_tool(mode: str) -> SyncTool:
    if mode == "rsync":
        tool = _find_rsync_tool()
        if tool is None:
            raise RuntimeError("rsync was requested but no local rsync executable was found")
        return tool
    if mode == "scp":
        tool = _find_scp_tool()
        if tool is None:
            raise RuntimeError("scp was requested but no local scp executable was found")
        return tool

    tool = _find_rsync_tool()
    if tool is not None:
        return tool
    tool = _find_scp_tool()
    if tool is not None:
        return tool
    raise RuntimeError("No supported transfer tool found. Install rsync or OpenSSH scp.")


def _remote_source(remote_root: str, subpath: PurePosixPath | None, *, for_rsync: bool) -> str:
    base = PurePosixPath(remote_root)
    source = base / subpath if subpath is not None else base
    text = source.as_posix()
    if for_rsync:
        return text.rstrip("/") + "/"
    return text.rstrip("/")


def _local_target(local_root: Path, subpath: PurePosixPath | None) -> Path:
    if subpath is None:
        return local_root
    return local_root.joinpath(*subpath.parts)


def _build_rsync_command(
    tool: SyncTool,
    *,
    remote: str,
    remote_root: str,
    local_root: Path,
    subpath: PurePosixPath | None,
    dry_run: bool,
    delete: bool,
) -> list[str]:
    target = _local_target(local_root, subpath)
    target.mkdir(parents=True, exist_ok=True)
    if tool.path_style == "wsl":
        local_arg = _to_wsl_path(target)
    elif tool.path_style == "msys":
        local_arg = _to_msys_path(target)
    else:
        local_arg = str(target)
    command = [*tool.command, "-avz", "--partial", "--progress", "--update", "--rsync-path=/usr/bin/rsync"]
    if dry_run:
        command.append("--dry-run")
    if delete:
        command.append("--delete")
    command.extend(
        [
            f"{remote}:{_remote_source(remote_root, subpath, for_rsync=True)}",
            local_arg,
        ]
    )
    return command


def _build_scp_command(
    tool: SyncTool,
    *,
    remote: str,
    remote_root: str,
    local_root: Path,
    subpath: PurePosixPath | None,
    dry_run: bool,
    delete: bool,
) -> list[str]:
    if dry_run:
        raise RuntimeError("--dry-run is only supported with rsync")
    if delete:
        raise RuntimeError("--delete is only supported with rsync")

    target = _local_target(local_root, subpath)
    if subpath is None:
        target.mkdir(parents=True, exist_ok=True)
        return [
            *tool.command,
            "-r",
            f"{remote}:{_remote_source(remote_root, subpath, for_rsync=False)}/.",
            str(target),
        ]

    target.parent.mkdir(parents=True, exist_ok=True)
    return [
        *tool.command,
        "-r",
        f"{remote}:{_remote_source(remote_root, subpath, for_rsync=False)}",
        str(target.parent),
    ]


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _subprocess_env_for_tool(tool: SyncTool) -> dict[str, str] | None:
    if tool.transfer_family != "rsync" or tool.path_style != "msys":
        return None

    env = os.environ.copy()
    tool_dir = str(Path(tool.command[0]).resolve().parent)
    env["PATH"] = tool_dir + os.pathsep + env.get("PATH", "")
    env["MSYS2_ARG_CONV_EXCL"] = "*"

    ssh_tool = _find_ssh_tool()
    if ssh_tool is not None:
        env["RSYNC_RSH"] = _to_msys_path(Path(ssh_tool.command[0]))
    return env


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f"{path.name}.tmp"
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("Ignoring unreadable JSON file at %s", path)
        return None
    return payload if isinstance(payload, dict) else None


def _lake_root_for_local_root(local_root: Path) -> Path:
    return local_root.parent if local_root.name == "l2_book" else local_root


def _discover_local_days(local_root: Path) -> list[str]:
    if not local_root.is_dir():
        return []
    days: list[str] = []
    for child in sorted(local_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("date="):
            continue
        days.append(child.name.removeprefix("date="))
    return days


def _prune_local_partitions(local_root: Path, min_date: str | None) -> list[str]:
    if min_date is None or not local_root.is_dir():
        return []

    removed_days: list[str] = []
    for child in sorted(local_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("date="):
            continue
        day = child.name.removeprefix("date=")
        if day >= min_date:
            continue
        shutil.rmtree(child)
        removed_days.append(day)
    return removed_days


def _list_local_parquet_files(local_root: Path) -> list[Path]:
    if not local_root.is_dir():
        return []
    return sorted(path for path in local_root.rglob("*.parquet") if path.is_file())


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _latest_local_parquet_file(parquet_files: list[Path], lake_root: Path) -> str | None:
    if not parquet_files:
        return None
    return _relative_path(parquet_files[-1], lake_root)


def _load_local_handoff(local_root: Path, lake_root: Path) -> dict[str, Any] | None:
    handoff_path = local_root / HANDOFF_RELATIVE_PATH
    payload = _read_json(handoff_path)
    if payload is None:
        return None
    payload = dict(payload)
    payload["path"] = _relative_path(handoff_path, lake_root)
    return payload


def _read_remote_file_snapshots(
    *,
    remote: str,
    remote_root: str,
    subpath: PurePosixPath | None,
    ssh_tool: SyncTool,
) -> list[RemoteFileSnapshot]:
    remote_path = _remote_source(remote_root, subpath, for_rsync=False)
    script = """
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
entries = []
if root.exists():
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".parquet" or rel == "_state/writer_handoff.json":
            stat = path.stat()
            entries.append({"path": rel, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
print(json.dumps(entries))
""".strip()
    remote_command = f"python3 -c {shlex.quote(script)} {shlex.quote(remote_path)}"
    command = [*ssh_tool.command, remote, remote_command]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"Remote manifest query failed: {message}")

    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Remote manifest query returned invalid JSON") from exc
    if not isinstance(payload, list):
        raise RuntimeError("Remote manifest query returned an invalid payload")

    snapshots: list[RemoteFileSnapshot] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        relative_path = item.get("path")
        size = item.get("size")
        mtime_ns = item.get("mtime_ns")
        if not isinstance(relative_path, str) or not isinstance(size, int) or not isinstance(mtime_ns, int):
            continue
        snapshots.append(RemoteFileSnapshot(relative_path=PurePosixPath(relative_path), size=size, mtime_ns=mtime_ns))
    return snapshots


def _plan_delta_scp_transfers(
    snapshots: list[RemoteFileSnapshot],
    *,
    local_root: Path,
) -> tuple[list[PurePosixPath], list[PurePosixPath]]:
    changed_directories: set[PurePosixPath] = set()
    changed_files: set[PurePosixPath] = set()

    for snapshot in snapshots:
        local_path = local_root.joinpath(*snapshot.relative_path.parts)
        if local_path.is_file():
            stat = local_path.stat()
            if stat.st_size == snapshot.size and stat.st_mtime_ns >= snapshot.mtime_ns:
                continue

        if snapshot.relative_path.suffix == ".parquet":
            changed_directories.add(snapshot.relative_path.parent)
        else:
            changed_files.add(snapshot.relative_path)

    return sorted(changed_directories), sorted(changed_files)


def _run_delta_scp_fallback(args: argparse.Namespace, subpath: PurePosixPath | None) -> dict[str, Any]:
    if args.dry_run:
        raise RuntimeError("delta scp fallback does not support --dry-run")
    if args.delete:
        raise RuntimeError("delta scp fallback does not support --delete")

    scp_tool = _choose_tool("scp")
    ssh_tool = _find_ssh_tool()
    if ssh_tool is None:
        raise RuntimeError("delta scp fallback requires a local ssh executable")

    local_root = args.local_root.resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    snapshots = _read_remote_file_snapshots(
        remote=args.remote,
        remote_root=args.remote_root,
        subpath=subpath,
        ssh_tool=ssh_tool,
    )
    changed_directories, changed_files = _plan_delta_scp_transfers(snapshots, local_root=local_root)
    base_remote = PurePosixPath(_remote_source(args.remote_root, subpath, for_rsync=False))

    started_at = _now_iso()
    started_clock = time.monotonic()

    for relative_dir in changed_directories:
        remote_dir = (base_remote / relative_dir).as_posix()
        local_parent = local_root if relative_dir.parent == PurePosixPath(".") else local_root.joinpath(*relative_dir.parent.parts)
        local_parent.mkdir(parents=True, exist_ok=True)
        command = [*scp_tool.command, "-r", f"{args.remote}:{remote_dir}", str(local_parent)]
        logging.info("Delta fallback copying changed directory | command=%s", _command_text(command))
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"delta scp fallback failed with exit code {completed.returncode}")

    for relative_file in changed_files:
        remote_file = (base_remote / relative_file).as_posix()
        local_parent = local_root if relative_file.parent == PurePosixPath(".") else local_root.joinpath(*relative_file.parent.parts)
        local_parent.mkdir(parents=True, exist_ok=True)
        command = [*scp_tool.command, f"{args.remote}:{remote_file}", str(local_parent)]
        logging.info("Delta fallback copying changed file | command=%s", _command_text(command))
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"delta scp fallback failed with exit code {completed.returncode}")

    duration_seconds = time.monotonic() - started_clock
    logging.info(
        "Delta fallback completed | changed_directories=%s | changed_files=%s",
        len(changed_directories),
        len(changed_files),
    )
    return {
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "transfer_tool": "delta-scp",
        "command": [
            scp_tool.command[0],
            "delta-fallback",
            f"changed_directories={len(changed_directories)}",
            f"changed_files={len(changed_files)}",
        ],
        "local_root": str(local_root),
    }


def _build_manifest_payload(
    *,
    remote: str,
    remote_root: str,
    local_root: Path,
    lake_root: Path,
    transfer_tool: str,
    transfer_command: list[str],
    min_date: str | None,
    started_at: str,
    duration_seconds: float,
    interval_seconds: float | None,
    subpath: str | None,
    pruned_days: list[str],
    enrichment_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    parquet_files = _list_local_parquet_files(local_root)
    days = _discover_local_days(local_root)
    total_bytes = sum(path.stat().st_size for path in parquet_files)
    local_handoff = _load_local_handoff(local_root, lake_root)
    market_count = enrichment_payload.get("market_count") if isinstance(enrichment_payload, dict) else None

    return {
        "generated_at": _now_iso(),
        "mode": "rolling_vps_sync",
        "remote": remote,
        "remote_root": remote_root,
        "local_root": str(local_root.resolve()),
        "lake_root": str(lake_root.resolve()),
        "subpath": subpath,
        "depth_levels": DEPTH_LEVELS,
        "days": days,
        "strict_schema": STRICT_SCHEMA,
        "scan_schema": SCAN_SCHEMA,
        "stats": {
            "day_count": len(days),
            "parquet_file_count": len(parquet_files),
            "parquet_bytes": total_bytes,
            "scan_column_count": len(SCAN_SCHEMA),
            "base_column_count": len(STRICT_SCHEMA),
            "market_count": market_count,
            "latest_parquet_file": _latest_local_parquet_file(parquet_files, lake_root),
            "handoff_present": local_handoff is not None,
        },
        "current_run": {
            "started_at": started_at,
            "completed_at": _now_iso(),
            "duration_seconds": round(duration_seconds, 3),
            "transfer_tool": transfer_tool,
            "transfer_command": _command_text(transfer_command),
            "loop_interval_seconds": interval_seconds,
            "min_date": min_date,
            "pruned_days": pruned_days,
            "handoff_state": local_handoff,
            "enriched_manifest": str((lake_root / ENRICHED_MANIFEST_NAME).resolve()) if enrichment_payload is not None else None,
        },
    }


def _build_sync_state_payload(
    *,
    remote: str,
    remote_root: str,
    local_root: Path,
    lake_root: Path,
    transfer_tool: str,
    min_date: str | None,
    interval_seconds: float | None,
    started_at: str,
    duration_seconds: float,
    enrichment_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    parquet_files = _list_local_parquet_files(local_root)
    last_successful_sync_at = _now_iso()
    return {
        "schema": "rolling_lake_sync_state_v1",
        "generated_at": _now_iso(),
        "last_started_at": started_at,
        "last_successful_sync_at": last_successful_sync_at,
        "last_duration_seconds": round(duration_seconds, 3),
        "remote": remote,
        "remote_root": remote_root,
        "local_root": str(local_root.resolve()),
        "lake_root": str(lake_root.resolve()),
        "transfer_tool": transfer_tool,
        "interval_seconds": interval_seconds,
        "min_date": min_date,
        "days": _discover_local_days(local_root),
        "latest_parquet_file": _latest_local_parquet_file(parquet_files, lake_root),
        "handoff_state": _load_local_handoff(local_root, lake_root),
        "market_count": enrichment_payload.get("market_count") if isinstance(enrichment_payload, dict) else None,
    }


def _refresh_lake_metadata(
    *,
    local_root: Path,
    remote: str,
    remote_root: str,
    transfer_tool: str,
    transfer_command: list[str],
    min_date: str | None,
    started_at: str,
    duration_seconds: float,
    interval_seconds: float | None,
    subpath: str | None,
    gamma_batch_size: int,
    gamma_timeout_seconds: float,
    fetcher: Callable[[list[str]], dict[str, GammaMarketRow]] | None = None,
) -> dict[str, Any]:
    local_root = local_root.resolve()
    lake_root = _lake_root_for_local_root(local_root)
    lake_root.mkdir(parents=True, exist_ok=True)

    pruned_days = _prune_local_partitions(local_root, min_date)
    initial_manifest = _build_manifest_payload(
        remote=remote,
        remote_root=remote_root,
        local_root=local_root,
        lake_root=lake_root,
        transfer_tool=transfer_tool,
        transfer_command=transfer_command,
        min_date=min_date,
        started_at=started_at,
        duration_seconds=duration_seconds,
        interval_seconds=interval_seconds,
        subpath=subpath,
        pruned_days=pruned_days,
        enrichment_payload=None,
    )
    _write_json_atomic(lake_root / MANIFEST_NAME, initial_manifest)

    enrichment_payload: dict[str, Any] | None = None
    parquet_files = _list_local_parquet_files(local_root)
    if parquet_files:
        enrichment_payload = run_enrichment(
            lake_root,
            gamma_batch_size=gamma_batch_size,
            timeout_seconds=gamma_timeout_seconds,
            fetcher=fetcher,
        )
    else:
        (lake_root / ENRICHED_MANIFEST_NAME).unlink(missing_ok=True)

    manifest_payload = _build_manifest_payload(
        remote=remote,
        remote_root=remote_root,
        local_root=local_root,
        lake_root=lake_root,
        transfer_tool=transfer_tool,
        transfer_command=transfer_command,
        min_date=min_date,
        started_at=started_at,
        duration_seconds=duration_seconds,
        interval_seconds=interval_seconds,
        subpath=subpath,
        pruned_days=pruned_days,
        enrichment_payload=enrichment_payload,
    )
    _write_json_atomic(lake_root / MANIFEST_NAME, manifest_payload)

    sync_state_payload = _build_sync_state_payload(
        remote=remote,
        remote_root=remote_root,
        local_root=local_root,
        lake_root=lake_root,
        transfer_tool=transfer_tool,
        min_date=min_date,
        interval_seconds=interval_seconds,
        started_at=started_at,
        duration_seconds=duration_seconds,
        enrichment_payload=enrichment_payload,
    )
    _write_json_atomic(lake_root / SYNC_STATE_NAME, sync_state_payload)

    return {
        "manifest": manifest_payload,
        "sync_state": sync_state_payload,
        "enriched_manifest": enrichment_payload,
        "manifest_path": lake_root / MANIFEST_NAME,
        "enriched_manifest_path": lake_root / ENRICHED_MANIFEST_NAME,
        "sync_state_path": lake_root / SYNC_STATE_NAME,
    }


def _run_transfer_iteration(args: argparse.Namespace, tool: SyncTool, subpath: PurePosixPath | None) -> dict[str, Any]:
    local_root = args.local_root.resolve()
    used_tool_name = tool.name
    if tool.transfer_family == "rsync":
        command = _build_rsync_command(
            tool,
            remote=args.remote,
            remote_root=args.remote_root,
            local_root=local_root,
            subpath=subpath,
            dry_run=bool(args.dry_run),
            delete=bool(args.delete),
        )
    else:
        command = _build_scp_command(
            tool,
            remote=args.remote,
            remote_root=args.remote_root,
            local_root=local_root,
            subpath=subpath,
            dry_run=bool(args.dry_run),
            delete=bool(args.delete),
        )

    started_at = _now_iso()
    started_clock = time.monotonic()
    logging.info("Rolling sync starting | tool=%s | command=%s", tool.name, _command_text(command))
    completed = subprocess.run(command, check=False, env=_subprocess_env_for_tool(tool))
    duration_seconds = time.monotonic() - started_clock
    if completed.returncode != 0:
        if tool.transfer_family == "rsync" and args.tool == "auto" and not args.dry_run and not args.delete:
            logging.warning(
                "Rsync transfer failed with exit code %s; falling back to delta scp for this iteration",
                completed.returncode,
            )
            fallback = _run_delta_scp_fallback(args, subpath)
            started_at = fallback["started_at"]
            duration_seconds = float(fallback["duration_seconds"])
            command = fallback["command"]
            used_tool_name = fallback["transfer_tool"]
        else:
            raise RuntimeError(f"Transfer command failed with exit code {completed.returncode}")

    metadata_result = None
    if not args.dry_run:
        metadata_result = _refresh_lake_metadata(
            local_root=local_root,
            remote=args.remote,
            remote_root=args.remote_root,
            transfer_tool=used_tool_name,
            transfer_command=command,
            min_date=_normalize_min_date(args.min_date),
            started_at=started_at,
            duration_seconds=duration_seconds,
            interval_seconds=float(args.interval_seconds) if args.loop else None,
            subpath=subpath.as_posix() if subpath is not None else None,
            gamma_batch_size=int(args.gamma_batch_size),
            gamma_timeout_seconds=float(args.gamma_timeout_seconds),
        )
        logging.info(
            "Rolling sync refreshed local lake metadata | manifest=%s | enriched_manifest=%s",
            metadata_result["manifest_path"],
            metadata_result["enriched_manifest_path"],
        )

    return {
        "started_at": started_at,
        "duration_seconds": round(duration_seconds, 3),
        "transfer_tool": used_tool_name,
        "command": command,
        "metadata": metadata_result,
        "local_root": str(local_root),
        "lake_root": str(_lake_root_for_local_root(local_root)),
    }


def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level)
    subpath = _clean_subpath(args.subpath)
    min_date = _normalize_min_date(args.min_date)
    if args.loop and args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive when --loop is set")

    tool = _choose_tool(args.tool)
    iteration = 0

    while True:
        try:
            result = _run_transfer_iteration(args, tool, subpath)
        except KeyboardInterrupt:
            logging.info("Rolling sync interrupted by operator")
            return 130
        except Exception as exc:
            logging.exception("Rolling sync iteration failed: %s", exc)
            if not args.loop:
                return 1
            logging.info("Retrying after %.1f seconds", float(args.interval_seconds))
            time.sleep(float(args.interval_seconds))
            continue

        if not args.loop:
            payload = {
                "transfer_tool": result["transfer_tool"],
                "command": _command_text(result["command"]),
                "local_root": result["local_root"],
                "lake_root": result["lake_root"],
                "min_date": min_date,
                "duration_seconds": result["duration_seconds"],
                "manifest": str((Path(result["lake_root"]) / MANIFEST_NAME).resolve()) if not args.dry_run else None,
                "enriched_manifest": str((Path(result["lake_root"]) / ENRICHED_MANIFEST_NAME).resolve()) if not args.dry_run else None,
                "sync_state": str((Path(result["lake_root"]) / SYNC_STATE_NAME).resolve()) if not args.dry_run else None,
                "stats": result["metadata"]["manifest"]["stats"] if result["metadata"] is not None else None,
            }
            print(json.dumps(payload, indent=2))
            return 0

        iteration += 1
        logging.info("Rolling sync iteration %s complete | sleeping %.1f seconds", iteration, float(args.interval_seconds))
        time.sleep(float(args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())