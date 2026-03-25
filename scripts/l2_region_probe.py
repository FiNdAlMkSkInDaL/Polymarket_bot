#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
import websockets.exceptions


DEFAULT_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass(frozen=True)
class SummaryStats:
    count: int
    mean: float | None
    stdev: float | None
    minimum: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    maximum: float | None


@dataclass(frozen=True)
class ProbeSummary:
    label: str
    ws_url: str
    channel: str
    asset_ids: list[str]
    started_at: str
    ended_at: str
    duration_s: float
    connect_attempts: int
    successful_connections: int
    disconnect_count: int
    total_frames: int
    total_events: int
    silence_gap_threshold_ms: float
    silence_gap_count: int
    max_silence_gap_ms: float
    connection_uptime_s: SummaryStats
    frame_gap_ms: SummaryStats
    exchange_lag_ms: SummaryStats
    per_event_type: dict[str, int]
    disconnect_reasons: dict[str, int]
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Polymarket L2/WebSocket feed stability for a fixed set of asset ids. "
            "This probe is standalone so it can run on minimal regional instances."
        )
    )
    parser.add_argument("--label", required=True, help="Human-readable region label.")
    parser.add_argument(
        "--asset-id",
        action="append",
        dest="asset_ids",
        default=[],
        help="Asset id to subscribe to. Repeatable.",
    )
    parser.add_argument(
        "--assets-file",
        default=None,
        help="Path to a JSON list or newline-delimited text file of asset ids.",
    )
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--channel", default="book")
    parser.add_argument("--ping-interval", type=float, default=20.0)
    parser.add_argument("--ping-timeout", type=float, default=20.0)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--max-size", type=int, default=2**24)
    parser.add_argument("--silence-gap-ms", type=float, default=1500.0)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=1,
        help="Ignore the first N received frames for jitter/lag stats to avoid subscription bootstrap distortion.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON summary.",
    )
    parser.add_argument(
        "--raw-sample-output",
        default=None,
        help="Optional path to write a small JSON sample of first-seen frames.",
    )
    parser.add_argument(
        "--raw-sample-limit",
        type=int,
        default=10,
        help="Maximum number of raw frames to preserve when --raw-sample-output is used.",
    )
    return parser.parse_args()


def _load_asset_ids(args: argparse.Namespace) -> list[str]:
    asset_ids = list(args.asset_ids)
    if args.assets_file:
        path = Path(args.assets_file)
        text = path.read_text(encoding="utf-8").strip()
        if text:
            if text.startswith("["):
                loaded = json.loads(text)
                if not isinstance(loaded, list):
                    raise SystemExit("assets file JSON must decode to a list of asset ids")
                asset_ids.extend(str(item).strip() for item in loaded)
            else:
                asset_ids.extend(line.strip() for line in text.splitlines() if line.strip())
    normalized = [asset_id for asset_id in dict.fromkeys(asset_ids) if asset_id]
    if not normalized:
        raise SystemExit("At least one --asset-id or --assets-file entry is required")
    return normalized


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires a non-empty list")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return float(lower + (upper - lower) * (rank - lower_index))


def summarize(values: list[float]) -> SummaryStats:
    if not values:
        return SummaryStats(0, None, None, None, None, None, None, None)
    ordered = sorted(values)
    mean = statistics.fmean(ordered)
    stdev = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return SummaryStats(
        count=len(ordered),
        mean=mean,
        stdev=stdev,
        minimum=ordered[0],
        p50=_percentile(ordered, 0.50),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
        maximum=ordered[-1],
    )


def extract_event_payloads(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def parse_timestamp_ms(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return None
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if numeric < 1e11:
        return numeric * 1000.0
    return numeric


class L2RegionProbe:
    def __init__(self, args: argparse.Namespace, asset_ids: list[str]) -> None:
        self.args = args
        self.asset_ids = asset_ids
        self.started_at = _utc_now_iso()
        self._started_monotonic = time.perf_counter()
        self._frame_gaps_ms: list[float] = []
        self._exchange_lag_ms: list[float] = []
        self._uptimes_s: list[float] = []
        self._event_type_counter: Counter[str] = Counter()
        self._disconnect_reasons: Counter[str] = Counter()
        self._total_frames = 0
        self._total_events = 0
        self._connect_attempts = 0
        self._successful_connections = 0
        self._disconnect_count = 0
        self._silence_gap_count = 0
        self._max_silence_gap_ms = 0.0
        self._last_frame_at: float | None = None
        self._frames_seen = 0
        self._sample_frames: list[dict[str, Any]] = []
        self._notes: list[str] = []

    async def run(self) -> ProbeSummary:
        deadline = self._started_monotonic + max(self.args.duration_s, 0.0)
        while time.perf_counter() < deadline:
            self._connect_attempts += 1
            connection_started = time.perf_counter()
            connected = False
            try:
                async with websockets.connect(
                    self.args.ws_url,
                    ping_interval=self.args.ping_interval,
                    ping_timeout=self.args.ping_timeout,
                    open_timeout=self.args.connect_timeout,
                    max_size=self.args.max_size,
                ) as websocket:
                    connected = True
                    self._successful_connections += 1
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "subscribe",
                                "channel": self.args.channel,
                                "assets_ids": self.asset_ids,
                            }
                        )
                    )
                    await self._consume_connection(websocket, deadline)
            except asyncio.TimeoutError:
                self._disconnect_count += 1
                self._disconnect_reasons["recv_timeout"] += 1
            except websockets.exceptions.ConnectionClosed as exc:
                self._disconnect_count += 1
                code = getattr(exc, "code", None)
                reason_key = f"connection_closed:{code}" if code is not None else "connection_closed"
                self._disconnect_reasons[reason_key] += 1
                self._uptimes_s.append(time.perf_counter() - connection_started)
            except OSError as exc:
                self._disconnect_count += 1
                self._disconnect_reasons[f"os_error:{type(exc).__name__}"] += 1
            except Exception as exc:
                self._disconnect_count += 1
                self._disconnect_reasons[f"unexpected:{type(exc).__name__}"] += 1
                self._notes.append(f"unexpected exception: {type(exc).__name__}: {exc}")
            finally:
                if connected:
                    self._uptimes_s.append(time.perf_counter() - connection_started)
            await asyncio.sleep(0.25)

        ended_at = _utc_now_iso()
        if not self._exchange_lag_ms:
            self._notes.append(
                "No exchange timestamps were observed in received payloads; ranking will rely on frame gaps and disconnects."
            )
        elif summarize(self._exchange_lag_ms).p50 is not None and summarize(self._exchange_lag_ms).p50 < 0:
            self._notes.append(
                "Median exchange lag is negative; absolute lag levels may reflect host clock skew or upstream timestamp semantics. Compare variance more heavily than the mean."
            )
        if self._successful_connections == 0:
            self._notes.append("No successful websocket connections were established.")
        exchange_lag_summary = summarize(self._exchange_lag_ms)
        return ProbeSummary(
            label=self.args.label,
            ws_url=self.args.ws_url,
            channel=self.args.channel,
            asset_ids=self.asset_ids,
            started_at=self.started_at,
            ended_at=ended_at,
            duration_s=time.perf_counter() - self._started_monotonic,
            connect_attempts=self._connect_attempts,
            successful_connections=self._successful_connections,
            disconnect_count=self._disconnect_count,
            total_frames=self._total_frames,
            total_events=self._total_events,
            silence_gap_threshold_ms=self.args.silence_gap_ms,
            silence_gap_count=self._silence_gap_count,
            max_silence_gap_ms=self._max_silence_gap_ms,
            connection_uptime_s=summarize(self._uptimes_s),
            frame_gap_ms=summarize(self._frame_gaps_ms),
            exchange_lag_ms=exchange_lag_summary,
            per_event_type=dict(sorted(self._event_type_counter.items())),
            disconnect_reasons=dict(sorted(self._disconnect_reasons.items())),
            notes=self._notes,
        )

    async def _consume_connection(self, websocket: Any, deadline: float) -> None:
        while time.perf_counter() < deadline:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=min(1.0, remaining))
            except asyncio.TimeoutError:
                if time.perf_counter() >= deadline:
                    return
                continue
            received_wall_ms = time.time() * 1000.0
            received_monotonic = time.perf_counter()
            self._frames_seen += 1
            self._record_frame_gap(received_monotonic)
            self._total_frames += 1

            if self.args.raw_sample_output and len(self._sample_frames) < self.args.raw_sample_limit:
                self._sample_frames.append(
                    {
                        "received_at": datetime.fromtimestamp(
                            received_wall_ms / 1000.0, tz=timezone.utc
                        ).isoformat().replace("+00:00", "Z"),
                        "raw": raw,
                    }
                )

            payload = json.loads(raw)
            event_payloads = extract_event_payloads(payload)
            self._total_events += len(event_payloads)
            for event_payload in event_payloads:
                event_type = str(event_payload.get("event_type") or event_payload.get("type") or "unknown")
                self._event_type_counter[event_type] += 1
                if self._frames_seen <= self.args.warmup_frames:
                    continue
                if event_type in {"book", "snapshot", "book_snapshot"}:
                    continue
                timestamp_ms = parse_timestamp_ms(event_payload.get("timestamp"))
                if timestamp_ms is None:
                    continue
                lag_ms = received_wall_ms - timestamp_ms
                if math.isfinite(lag_ms):
                    self._exchange_lag_ms.append(lag_ms)

    def _record_frame_gap(self, received_monotonic: float) -> None:
        if self._frames_seen <= self.args.warmup_frames:
            self._last_frame_at = received_monotonic
            return
        if self._last_frame_at is not None:
            gap_ms = (received_monotonic - self._last_frame_at) * 1000.0
            if math.isfinite(gap_ms):
                self._frame_gaps_ms.append(gap_ms)
                self._max_silence_gap_ms = max(self._max_silence_gap_ms, gap_ms)
                if gap_ms >= self.args.silence_gap_ms:
                    self._silence_gap_count += 1
        self._last_frame_at = received_monotonic

    def write_outputs(self, summary: ProbeSummary) -> None:
        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        if self.args.raw_sample_output:
            raw_output_path = Path(self.args.raw_sample_output)
            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_output_path.write_text(json.dumps(self._sample_frames, indent=2), encoding="utf-8")


async def _async_main() -> int:
    args = parse_args()
    asset_ids = _load_asset_ids(args)
    probe = L2RegionProbe(args, asset_ids)
    summary = await probe.run()
    probe.write_outputs(summary)
    print(json.dumps(asdict(summary), indent=2))
    return 0


def main() -> int:
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())