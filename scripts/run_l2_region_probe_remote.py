#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegionHost:
    label: str
    host: str
    remote_dir: str
    python_bin: str = "python3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy and run the standalone L2 region probe over SSH for multiple regional hosts."
        )
    )
    parser.add_argument("--inventory", required=True, help="JSON file describing remote hosts.")
    parser.add_argument("--output-dir", required=True, help="Local directory for pulled summaries.")
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--ws-url", default="wss://ws-subscriptions-clob.polymarket.com/ws/market")
    parser.add_argument("--channel", default="book")
    parser.add_argument("--ping-interval", type=float, default=20.0)
    parser.add_argument("--ping-timeout", type=float, default=20.0)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--silence-gap-ms", type=float, default=1500.0)
    parser.add_argument("--asset-id", action="append", dest="asset_ids", default=[])
    parser.add_argument("--assets-file", default=None)
    return parser.parse_args()


def load_inventory(path: Path) -> list[RegionHost]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("inventory JSON must decode to a list")
    hosts: list[RegionHost] = []
    for row in raw:
        if not isinstance(row, dict):
            raise SystemExit("inventory rows must be objects")
        hosts.append(
            RegionHost(
                label=str(row["label"]),
                host=str(row["host"]),
                remote_dir=str(row.get("remote_dir", "~/l2-region-probe")),
                python_bin=str(row.get("python_bin", "python3")),
            )
        )
    return hosts


async def run_command(command: list[str]) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return process.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode(
        "utf-8", errors="replace"
    )


def build_remote_probe_args(args: argparse.Namespace, region: RegionHost, remote_assets_file: str | None) -> list[str]:
    probe_args = [
        ".venv/bin/python",
        "l2_region_probe.py",
        "--label",
        region.label,
        "--duration-s",
        str(args.duration_s),
        "--ws-url",
        args.ws_url,
        "--channel",
        args.channel,
        "--ping-interval",
        str(args.ping_interval),
        "--ping-timeout",
        str(args.ping_timeout),
        "--connect-timeout",
        str(args.connect_timeout),
        "--silence-gap-ms",
        str(args.silence_gap_ms),
        "--output",
        f"summary_{region.label}.json",
    ]
    for asset_id in args.asset_ids:
        probe_args.extend(["--asset-id", asset_id])
    if remote_assets_file:
        probe_args.extend(["--assets-file", remote_assets_file])
    return probe_args


async def deploy_and_run_region(
    args: argparse.Namespace,
    region: RegionHost,
    probe_script: Path,
    assets_file: Path | None,
    output_dir: Path,
) -> dict[str, str | int]:
    remote_probe_path = f"{region.host}:{region.remote_dir.rstrip('/')}/l2_region_probe.py"
    remote_assets_name = None
    if assets_file is not None:
        remote_assets_name = assets_file.name
    mkdir_cmd = ["ssh", region.host, f"mkdir -p {shlex.quote(region.remote_dir)}"]
    code, stdout, stderr = await run_command(mkdir_cmd)
    if code != 0:
        return {"label": region.label, "status": "mkdir_failed", "details": stderr or stdout}

    copy_probe_cmd = ["scp", str(probe_script), remote_probe_path]
    code, stdout, stderr = await run_command(copy_probe_cmd)
    if code != 0:
        return {"label": region.label, "status": "copy_probe_failed", "details": stderr or stdout}

    if assets_file is not None:
        copy_assets_cmd = [
            "scp",
            str(assets_file),
            f"{region.host}:{region.remote_dir.rstrip('/')}/{assets_file.name}",
        ]
        code, stdout, stderr = await run_command(copy_assets_cmd)
        if code != 0:
            return {"label": region.label, "status": "copy_assets_failed", "details": stderr or stdout}

    bootstrap_cmd = (
        f"cd {shlex.quote(region.remote_dir)} && "
        f"if [ ! -d .venv ]; then {shlex.quote(region.python_bin)} -m venv .venv; fi && "
        ".venv/bin/python -m pip install --quiet --upgrade pip websockets"
    )
    code, stdout, stderr = await run_command(["ssh", region.host, bootstrap_cmd])
    if code != 0:
        return {"label": region.label, "status": "bootstrap_failed", "details": stderr or stdout}

    remote_probe_args = build_remote_probe_args(args, region, remote_assets_name)
    remote_probe_cmd = (
        f"cd {shlex.quote(region.remote_dir)} && "
        + " ".join(shlex.quote(part) for part in remote_probe_args)
    )
    code, stdout, stderr = await run_command(["ssh", region.host, remote_probe_cmd])
    if code != 0:
        return {"label": region.label, "status": "probe_failed", "details": stderr or stdout}

    local_output_path = output_dir / f"{region.label}.json"
    fetch_cmd = [
        "scp",
        f"{region.host}:{region.remote_dir.rstrip('/')}/summary_{region.label}.json",
        str(local_output_path),
    ]
    code, fetch_stdout, fetch_stderr = await run_command(fetch_cmd)
    if code != 0:
        return {
            "label": region.label,
            "status": "fetch_failed",
            "details": fetch_stderr or fetch_stdout,
        }
    return {"label": region.label, "status": "ok", "details": str(local_output_path)}


async def _async_main() -> int:
    args = parse_args()
    if not args.asset_ids and not args.assets_file:
        raise SystemExit("Provide at least one --asset-id or --assets-file")

    inventory = load_inventory(Path(args.inventory))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_script = Path(__file__).with_name("l2_region_probe.py")
    assets_file = Path(args.assets_file).resolve() if args.assets_file else None

    results = await asyncio.gather(
        *(deploy_and_run_region(args, region, probe_script, assets_file, output_dir) for region in inventory)
    )
    print(json.dumps(results, indent=2))
    return 0 if all(result["status"] == "ok" for result in results) else 1


def main() -> int:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())