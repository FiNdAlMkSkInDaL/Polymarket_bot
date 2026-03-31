#!/usr/bin/env python3
"""Generate reward-shadow summaries from persisted shadow trades."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.trade_store import TradeStore, create_wal_safe_remeasurement_snapshot  # noqa: E402


DEFAULT_DB = Path("logs/trades.db")
DEFAULT_SNAPSHOT_DIR = Path("artifacts/reward_shadow_snapshots")
REWARD_EXTRA_KEYS = {
    "reward_daily_usd",
    "reward_min_size",
    "reward_max_spread_cents",
    "competition_usd",
    "reward_to_competition",
    "queue_depth_ahead_usd",
    "queue_residency_seconds",
    "fill_occurred",
    "fill_latency_ms",
    "markout_5s_cents",
    "markout_15s_cents",
    "markout_60s_cents",
    "estimated_reward_capture_usd",
    "estimated_net_edge_usd",
    "quote_id",
    "quote_reason",
    "emergency_flatten",
}
REWARD_TO_COMP_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (-float("inf"), 0.25, "<0.25"),
    (0.25, 0.5, "0.25-0.50"),
    (0.5, 1.0, "0.50-1.00"),
    (1.0, 2.0, "1.00-2.00"),
    (2.0, float("inf"), ">=2.00"),
)


def _bucket_reward_to_competition(value: Any) -> str:
    if value is None:
        return "missing"
    numeric = float(value)
    for lower, upper, label in REWARD_TO_COMP_BUCKETS:
        if lower <= numeric < upper:
            return label
    return REWARD_TO_COMP_BUCKETS[-1][2]


def _bool_bucket(value: Any) -> str:
    if value is None:
        return "missing"
    return "true" if bool(value) else "false"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _extract_extra_payload(payload_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_json or "{}")
    except json.JSONDecodeError:
        return {}
    extra_payload = payload.get("extra_payload")
    return extra_payload if isinstance(extra_payload, dict) else {}


def _is_reward_shadow_row(extra_payload: dict[str, Any]) -> bool:
    return any(key in extra_payload for key in REWARD_EXTRA_KEYS)


async def load_reward_shadow_rows(
    *,
    db_path: Path,
    signal_source: str | None = None,
) -> list[dict[str, Any]]:
    store = TradeStore(db_path)
    try:
        await store.init()
        sql = (
            "SELECT s.id, s.signal_source, s.market_id, s.direction, s.entry_price, s.entry_size, "
            "s.entry_time, s.target_price, s.stop_price, s.exit_price, s.exit_time, s.exit_reason, "
            "s.pnl_cents, s.hold_seconds, s.confidence, s.zscore, j.payload_json "
            "FROM shadow_trades s "
            "LEFT JOIN trade_persistence_journal j ON j.journal_key = ('shadow_trades:' || s.id) "
            "WHERE s.state = ?"
        )
        params: list[object] = ["CLOSED"]
        if signal_source is not None:
            sql += " AND s.signal_source = ?"
            params.append(signal_source)
        sql += " ORDER BY s.exit_time ASC, s.id ASC"

        cursor = await store._db.execute(sql, tuple(params))
        raw_rows = await cursor.fetchall()
    finally:
        await store.close()

    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        extra_payload = _extract_extra_payload(str(raw_row[16] or "{}"))
        if not _is_reward_shadow_row(extra_payload):
            continue
        row = {
            "id": raw_row[0],
            "signal_source": raw_row[1],
            "market_id": raw_row[2],
            "direction": raw_row[3],
            "entry_price": float(raw_row[4] or 0.0),
            "entry_size": float(raw_row[5] or 0.0),
            "entry_time": float(raw_row[6] or 0.0),
            "target_price": float(raw_row[7] or 0.0),
            "stop_price": float(raw_row[8] or 0.0),
            "exit_price": float(raw_row[9] or 0.0),
            "exit_time": float(raw_row[10] or 0.0),
            "exit_reason": raw_row[11],
            "pnl_cents": float(raw_row[12] or 0.0),
            "hold_seconds": float(raw_row[13] or 0.0),
            "confidence": float(raw_row[14] or 0.0),
            "zscore": float(raw_row[15] or 0.0),
            "extra_payload": extra_payload,
            "reward_to_competition_bucket": _bucket_reward_to_competition(extra_payload.get("reward_to_competition")),
            "fill_occurred_bucket": _bool_bucket(extra_payload.get("fill_occurred")),
            "emergency_flatten_bucket": _bool_bucket(extra_payload.get("emergency_flatten")),
        }
        rows.append(row)
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_trades = len(rows)
    total_pnl_cents = sum(float(row.get("pnl_cents") or 0.0) for row in rows)
    reward_capture_values = [
        value
        for value in (_safe_float(row["extra_payload"].get("estimated_reward_capture_usd")) for row in rows)
        if value is not None
    ]
    net_edge_values = [
        value
        for value in (_safe_float(row["extra_payload"].get("estimated_net_edge_usd")) for row in rows)
        if value is not None
    ]
    fill_count = sum(1 for row in rows if row["extra_payload"].get("fill_occurred") is True)
    return {
        "total_trades": total_trades,
        "total_pnl_cents": round(total_pnl_cents, 2),
        "expectancy_cents": round(total_pnl_cents / total_trades, 2) if total_trades else 0.0,
        "fill_count": fill_count,
        "fill_rate": round(fill_count / total_trades, 4) if total_trades else 0.0,
        "avg_estimated_reward_capture_usd": round(sum(reward_capture_values) / len(reward_capture_values), 6)
        if reward_capture_values
        else None,
        "avg_estimated_net_edge_usd": round(sum(net_edge_values) / len(net_edge_values), 6)
        if net_edge_values
        else None,
    }


def _group_rows(rows: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key_name])].append(row)
    return {label: _summarize_rows(bucket_rows) for label, bucket_rows in sorted(grouped.items())}


def build_reward_shadow_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "summary": _summarize_rows(rows),
        "by_market": _group_rows(rows, "market_id"),
        "by_signal_source": _group_rows(rows, "signal_source"),
        "by_reward_to_competition_bucket": _group_rows(rows, "reward_to_competition_bucket"),
        "by_fill_occurred": _group_rows(rows, "fill_occurred_bucket"),
        "by_emergency_flatten": _group_rows(rows, "emergency_flatten_bucket"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Reward Shadow Report",
        "",
        "## Summary",
        "",
        f"- Trades: `{int(summary.get('total_trades', 0) or 0)}`",
        f"- Expectancy: `{float(summary.get('expectancy_cents', 0.0) or 0.0):+.2f}c/trade`",
        f"- Total PnL: `{float(summary.get('total_pnl_cents', 0.0) or 0.0):+.2f}c`",
        f"- Fill rate: `{float(summary.get('fill_rate', 0.0) or 0.0):.1%}`",
        f"- Avg estimated reward capture: `{summary.get('avg_estimated_reward_capture_usd')}`",
        f"- Avg estimated net edge: `{summary.get('avg_estimated_net_edge_usd')}`",
        "",
    ]
    for section_key, title in (
        ("by_market", "By Market"),
        ("by_signal_source", "By Signal Source"),
        ("by_reward_to_competition_bucket", "By Reward To Competition"),
        ("by_fill_occurred", "By Fill Occurred"),
        ("by_emergency_flatten", "By Emergency Flatten"),
    ):
        lines.extend([f"## {title}", ""])
        section = report.get(section_key, {})
        if not section:
            lines.append("- `none`: n=`0`")
            lines.append("")
            continue
        for label, stats in section.items():
            lines.append(
                f"- `{label}`: n=`{int(stats.get('total_trades', 0) or 0)}`, "
                f"fill_rate=`{float(stats.get('fill_rate', 0.0) or 0.0):.1%}`, "
                f"avg_reward=`{stats.get('avg_estimated_reward_capture_usd')}`, "
                f"avg_net=`{stats.get('avg_estimated_net_edge_usd')}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Build a reward-shadow report from persisted shadow trades")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to trades.db")
    parser.add_argument("--signal-source", type=str, default=None, help="Optional signal source filter")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Optional Markdown output path")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR, help="Directory for WAL-safe analysis snapshots")
    parser.add_argument("--no-snapshot", action="store_true", help="Read the database directly instead of creating a WAL-safe snapshot")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"Database not found: {args.db}")

    analysis_db = args.db
    if not args.no_snapshot:
        snapshot_manifest = await create_wal_safe_remeasurement_snapshot(
            db_path=args.db,
            label="reward_shadow_report",
            output_dir=args.snapshot_dir,
        )
        analysis_db = Path(str(snapshot_manifest["snapshot_db_path"]))

    rows = await load_reward_shadow_rows(db_path=analysis_db, signal_source=args.signal_source)
    report = build_reward_shadow_report(rows)
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(_main())