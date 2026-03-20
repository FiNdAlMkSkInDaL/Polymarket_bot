#!/usr/bin/env python3
"""Visualize L2 top-of-book and wick trades for a single Polymarket market.

This script reads raw JSONL or processed Parquet tick data, reconstructs the
best bid / best ask over time from snapshot + delta messages, extracts trades,
and highlights trades that executed more than a configurable percentage away
from the rolling 1-minute average mid-price.

Examples
--------
    python scripts/visualize_l2_wicks.py data/vps_march2026/ticks/2026-03-18 \
        0x06b066958f047d9f684a60ba923966ffd581bcda87436327d475bdfbd0d1d34f

    python scripts/visualize_l2_wicks.py data/vps_march2026_parquet/2026-03-18 \
        0x06b066958f047d9f684a60ba923966ffd581bcda87436327d475bdfbd0d1d34f \
        --output charts/l2_wicks.png --show
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


SNAPSHOT_TYPES = {"book", "snapshot", "book_snapshot"}
TRADE_TYPES = {"trade", "last_trade_price"}
DELTA_TYPES = {"price_change", "delta", "l2_delta"}


@dataclass(slots=True)
class EventRecord:
    timestamp: float
    event_type: str
    payload: dict[str, Any]
    market_id: str


@dataclass(slots=True)
class BookPoint:
    timestamp: float
    best_bid: float | None
    best_ask: float | None
    mid_price: float | None


@dataclass(slots=True)
class TradePoint:
    timestamp: float
    price: float
    size: float | None
    side: str | None
    rolling_mid_ma: float | None
    deviation_pct: float | None
    is_wick: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the L2 top-of-book for one market and highlight trades "
            "that execute far from the rolling 1-minute mid-price average."
        )
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help=(
            "Path to a raw tick date directory, a processed parquet date directory, "
            "or a parent directory that contains them."
        ),
    )
    parser.add_argument(
        "market_id",
        help="Hex market id to analyze, for example the JSONL file stem.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output chart path. Defaults to ./l2_wicks_<market_id-prefix>.png.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=60.0,
        help="Rolling window size for the moving-average mid-price baseline.",
    )
    parser.add_argument(
        "--wick-threshold-pct",
        type=float,
        default=5.0,
        help="Percent deviation from rolling mid-price MA required to flag a wick.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of discovered input files to scan.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the chart interactively after saving.",
    )
    return parser.parse_args(argv)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _normalize_timestamp(value: Any) -> float | None:
    ts = _safe_float(value)
    if ts is None:
        return None
    if ts > 1e15:
        ts /= 1_000_000.0
    elif ts > 1e12:
        ts /= 1_000.0
    return ts


def _resolve_market_id(top_level_asset_id: Any, payload: dict[str, Any]) -> str:
    payload_market = payload.get("market")
    if payload_market:
        return str(payload_market)
    if top_level_asset_id:
        return str(top_level_asset_id)
    payload_asset_id = payload.get("asset_id")
    if payload_asset_id:
        return str(payload_asset_id)
    return ""


def _record_matches_market(requested_market_id: str, top_level_asset_id: Any, payload: dict[str, Any]) -> bool:
    candidates = {
        str(value)
        for value in (
            top_level_asset_id,
            payload.get("market"),
            payload.get("asset_id"),
        )
        if value is not None and str(value)
    }
    return requested_market_id in candidates


def _classify_event(source: str, payload: dict[str, Any], parquet_msg_type: str | None = None) -> str | None:
    payload_type = str(payload.get("event_type", "") or "").lower()
    source = (source or "").lower()
    parquet_msg_type = (parquet_msg_type or "").lower()

    if payload_type in SNAPSHOT_TYPES or parquet_msg_type == "snapshot":
        return "snapshot"
    if payload_type in TRADE_TYPES or source == "trade" or parquet_msg_type == "trade":
        return "trade"
    if payload_type in DELTA_TYPES or source in {"l2", "l2_delta"} or parquet_msg_type == "delta":
        return "delta"
    return None


def discover_input_files(data_path: Path, market_id: str, max_files: int | None = None) -> list[Path]:
    if not data_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {data_path}")

    if data_path.is_file():
        if data_path.suffix not in {".jsonl", ".parquet"}:
            raise ValueError(f"Unsupported file type: {data_path.suffix}")
        return [data_path]

    jsonl_candidates = sorted(data_path.rglob(f"{market_id}.jsonl"))
    if jsonl_candidates:
        files = jsonl_candidates
    else:
        files = sorted(data_path.rglob("*.jsonl")) + sorted(data_path.rglob("*.parquet"))

    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise FileNotFoundError(f"No .jsonl or .parquet files found under {data_path}")

    return files


def _iter_jsonl_events(file_path: Path, market_id: str) -> Iterator[EventRecord]:
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(
                    f"Skipping malformed JSON in {file_path} line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue

            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue

            top_level_market_id = record.get("asset_id")
            if not _record_matches_market(market_id, top_level_market_id, payload):
                continue
            resolved_market_id = _resolve_market_id(top_level_market_id, payload)

            timestamp = _normalize_timestamp(record.get("local_ts"))
            if timestamp is None:
                continue

            event_type = _classify_event(str(record.get("source", "")), payload)
            if event_type is None:
                continue

            yield EventRecord(
                timestamp=timestamp,
                event_type=event_type,
                payload=payload,
                market_id=resolved_market_id,
            )


def _iter_parquet_events(file_path: Path, market_id: str) -> Iterator[EventRecord]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet files requires pyarrow. Install the optional data dependencies."
        ) from exc

    table = pq.read_table(str(file_path), columns=["local_ts", "msg_type", "asset_id", "payload"])
    local_ts_values = table.column("local_ts").to_pylist()
    msg_types = table.column("msg_type").to_pylist()
    asset_ids = table.column("asset_id").to_pylist()
    payload_values = table.column("payload").to_pylist()

    for index in range(table.num_rows):
        top_level_market_id = asset_ids[index]

        timestamp = _normalize_timestamp(local_ts_values[index])
        if timestamp is None:
            continue

        try:
            payload = json.loads(payload_values[index])
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(payload, dict):
            continue

        if not _record_matches_market(market_id, top_level_market_id, payload):
            continue
        resolved_market_id = _resolve_market_id(top_level_market_id, payload)

        event_type = _classify_event("", payload, parquet_msg_type=str(msg_types[index]))
        if event_type is None:
            continue

        yield EventRecord(
            timestamp=timestamp,
            event_type=event_type,
            payload=payload,
            market_id=resolved_market_id,
        )


def load_events(files: Iterable[Path], market_id: str) -> list[EventRecord]:
    events: list[EventRecord] = []
    for file_path in files:
        if file_path.suffix == ".jsonl":
            events.extend(_iter_jsonl_events(file_path, market_id))
        elif file_path.suffix == ".parquet":
            events.extend(_iter_parquet_events(file_path, market_id))
    events.sort(key=lambda event: event.timestamp)
    return events


def _best_bid(bids: dict[float, float]) -> float | None:
    return max(bids) if bids else None


def _best_ask(asks: dict[float, float]) -> float | None:
    return min(asks) if asks else None


def _changes_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    changes = payload.get("changes")
    if isinstance(changes, list):
        return [change for change in changes if isinstance(change, dict)]

    fallback_change = {
        "side": payload.get("side"),
        "price": payload.get("price") or payload.get("change_price"),
        "size": payload.get("size") or payload.get("change_size"),
    }
    if fallback_change["side"] is None and fallback_change["price"] is None:
        return []
    return [fallback_change]


def reconstruct_market_microstructure(
    events: Iterable[EventRecord],
    window_seconds: float,
    wick_threshold_pct: float,
) -> tuple[list[BookPoint], list[TradePoint]]:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    book_points: list[BookPoint] = []
    trade_points: list[TradePoint] = []
    rolling_mid_window: deque[tuple[float, float]] = deque()

    for event in events:
        if event.event_type == "snapshot":
            next_bids: dict[float, float] = {}
            next_asks: dict[float, float] = {}
            for level in event.payload.get("bids", []):
                if not isinstance(level, dict):
                    continue
                price = _safe_float(level.get("price"))
                size = _safe_float(level.get("size"))
                if price is None or size is None or size <= 0:
                    continue
                next_bids[price] = size
            for level in event.payload.get("asks", []):
                if not isinstance(level, dict):
                    continue
                price = _safe_float(level.get("price"))
                size = _safe_float(level.get("size"))
                if price is None or size is None or size <= 0:
                    continue
                next_asks[price] = size
            bids = next_bids
            asks = next_asks

        elif event.event_type == "delta":
            for change in _changes_from_payload(event.payload):
                side = str(change.get("side", "") or "").upper()
                price = _safe_float(change.get("price"))
                size = _safe_float(change.get("size"))
                if price is None:
                    continue
                book_side = bids if side == "BUY" else asks
                if size is None or size <= 0:
                    book_side.pop(price, None)
                else:
                    book_side[price] = size

        if event.event_type in {"snapshot", "delta"}:
            best_bid = _best_bid(bids)
            best_ask = _best_ask(asks)
            mid_price = None
            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2.0
                rolling_mid_window.append((event.timestamp, mid_price))
                cutoff = event.timestamp - window_seconds
                while rolling_mid_window and rolling_mid_window[0][0] < cutoff:
                    rolling_mid_window.popleft()
            book_points.append(
                BookPoint(
                    timestamp=event.timestamp,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mid_price=mid_price,
                )
            )
            continue

        if event.event_type != "trade":
            continue

        trade_price = _safe_float(event.payload.get("price"))
        if trade_price is None:
            continue
        trade_size = _safe_float(event.payload.get("size") or event.payload.get("amount"))
        trade_side = event.payload.get("side")

        cutoff = event.timestamp - window_seconds
        while rolling_mid_window and rolling_mid_window[0][0] < cutoff:
            rolling_mid_window.popleft()

        rolling_mid_ma = None
        if rolling_mid_window:
            rolling_mid_ma = sum(mid for _, mid in rolling_mid_window) / len(rolling_mid_window)

        deviation_pct = None
        is_wick = False
        if rolling_mid_ma is not None and rolling_mid_ma > 0:
            deviation_pct = abs(trade_price - rolling_mid_ma) / rolling_mid_ma * 100.0
            is_wick = deviation_pct > wick_threshold_pct

        trade_points.append(
            TradePoint(
                timestamp=event.timestamp,
                price=trade_price,
                size=trade_size,
                side=str(trade_side).lower() if trade_side is not None else None,
                rolling_mid_ma=rolling_mid_ma,
                deviation_pct=deviation_pct,
                is_wick=is_wick,
            )
        )

    return book_points, trade_points


def render_chart(
    book_points: list[BookPoint],
    trade_points: list[TradePoint],
    market_id: str,
    output_path: Path,
    wick_threshold_pct: float,
    show: bool,
) -> None:
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Rendering the chart requires matplotlib. Install the optional dev/data plotting dependencies."
        ) from exc

    if not book_points:
        raise RuntimeError("No L2 snapshot/delta events were found for the requested market.")
    if not trade_points:
        raise RuntimeError("No trade events were found for the requested market.")

    timestamps = [datetime.fromtimestamp(point.timestamp, tz=timezone.utc) for point in book_points]
    best_bids = [point.best_bid for point in book_points]
    best_asks = [point.best_ask for point in book_points]

    normal_trades = [trade for trade in trade_points if not trade.is_wick]
    wick_trades = [trade for trade in trade_points if trade.is_wick]
    buy_trades = [trade for trade in normal_trades if trade.side == "buy"]
    sell_trades = [trade for trade in normal_trades if trade.side == "sell"]
    unknown_side_trades = [trade for trade in normal_trades if trade.side not in {"buy", "sell"}]

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    ax.plot(timestamps, best_bids, color="#138f3e", linewidth=1.2, label="Best Bid")
    ax.plot(timestamps, best_asks, color="#cf2e2e", linewidth=1.2, label="Best Ask")

    if buy_trades:
        ax.scatter(
            [datetime.fromtimestamp(trade.timestamp, tz=timezone.utc) for trade in buy_trades],
            [trade.price for trade in buy_trades],
            color="#1f77b4",
            s=18,
            alpha=0.65,
            label="Trades (buy)",
        )
    if sell_trades:
        ax.scatter(
            [datetime.fromtimestamp(trade.timestamp, tz=timezone.utc) for trade in sell_trades],
            [trade.price for trade in sell_trades],
            color="#ff8c00",
            s=18,
            alpha=0.65,
            label="Trades (sell)",
        )
    if unknown_side_trades:
        ax.scatter(
            [datetime.fromtimestamp(trade.timestamp, tz=timezone.utc) for trade in unknown_side_trades],
            [trade.price for trade in unknown_side_trades],
            color="#606060",
            s=18,
            alpha=0.6,
            label="Trades (unknown side)",
        )
    if wick_trades:
        ax.scatter(
            [datetime.fromtimestamp(trade.timestamp, tz=timezone.utc) for trade in wick_trades],
            [trade.price for trade in wick_trades],
            color="#ffd23f",
            edgecolors="#121212",
            linewidths=0.8,
            marker="*",
            s=160,
            alpha=0.95,
            label=f"Wick trades (>{wick_threshold_pct:.1f}% from 1m MA mid)",
            zorder=5,
        )

    ax.set_title(f"L2 Wick Visualizer\n{market_id}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M", tz=timezone.utc))
    fig.autofmt_xdate(rotation=30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def print_summary(
    files: list[Path],
    book_points: list[BookPoint],
    trade_points: list[TradePoint],
    output_path: Path,
) -> None:
    wick_trades = [trade for trade in trade_points if trade.is_wick]
    print(f"Scanned files: {len(files)}")
    print(f"Book updates: {len(book_points)}")
    print(f"Trades: {len(trade_points)}")
    print(f"Wick trades: {len(wick_trades)}")
    if wick_trades:
        print("Top wick trades by deviation:")
        for trade in sorted(
            wick_trades,
            key=lambda item: item.deviation_pct or 0.0,
            reverse=True,
        )[:10]:
            trade_dt = datetime.fromtimestamp(trade.timestamp, tz=timezone.utc).isoformat()
            size_text = "?" if trade.size is None else f"{trade.size:.4f}"
            deviation_text = "?" if trade.deviation_pct is None else f"{trade.deviation_pct:.2f}%"
            baseline_text = "?" if trade.rolling_mid_ma is None else f"{trade.rolling_mid_ma:.4f}"
            print(
                f"  {trade_dt}  price={trade.price:.4f}  size={size_text}  "
                f"side={trade.side or 'unknown'}  baseline={baseline_text}  deviation={deviation_text}"
            )
    print(f"Saved chart to {output_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output or Path(f"l2_wicks_{args.market_id[:12]}.png")

    try:
        files = discover_input_files(args.data_path, args.market_id, max_files=args.max_files)
        events = load_events(files, args.market_id)
        if not events:
            raise RuntimeError(
                "No matching events found for the requested market_id. "
                "Check that the market id matches the JSONL file stem / parquet asset_id."
            )
        book_points, trade_points = reconstruct_market_microstructure(
            events,
            window_seconds=args.window_seconds,
            wick_threshold_pct=args.wick_threshold_pct,
        )
        render_chart(
            book_points,
            trade_points,
            args.market_id,
            output_path,
            args.wick_threshold_pct,
            args.show,
        )
        print_summary(files, book_points, trade_points, output_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())