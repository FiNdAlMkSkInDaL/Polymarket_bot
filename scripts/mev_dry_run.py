from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.execution.mev_dispatcher import MevDispatcher
from src.execution.mev_router import MevExecutionBatch, MevExecutionRouter
from src.execution.mev_serializer import serialize_mev_execution_batch
from mev_fixtures import (
    DEFAULT_ATTACK_VOLUME,
    DEFAULT_D3_MAX_CAPITAL,
    DEFAULT_D3_PANIC_LIMIT,
    DEFAULT_SHADOW_MAX_CAPITAL,
    DEFAULT_SHADOW_PREMIUM_PCT,
    SCENARIO_FIXTURES,
    build_dispute_signal,
    build_mm_signal,
    build_shadow_signal,
)


def _format_quantity(value: float) -> str:
    rounded_int = round(float(value))
    if abs(float(value) - rounded_int) <= 1e-4:
        return f"{rounded_int:,}"
    return f"{float(value):,.4f}".rstrip("0").rstrip(".")


def _format_currency(value: float) -> str:
    return f"${float(value):,.2f}"


def _round_price(value: float) -> float:
    return round(float(value), 2)


def _serialize_batch_payload(batch: MevExecutionBatch) -> dict[str, Any]:
    return json.loads(serialize_mev_execution_batch(batch))


def _shadow_clamp_active(fixture, batch: MevExecutionBatch) -> bool:
    maker_payload = batch.payloads[1]
    snapshot = fixture.snapshots[maker_payload.market_id]
    target_price = _round_price(
        snapshot.mid_price(maker_payload.direction) + float(maker_payload.metadata["premium_pct"])
    )
    return abs(maker_payload.price - target_price) > 1e-9


def _mm_trap_clamp_active(fixture, batch: MevExecutionBatch) -> bool:
    maker_payload = batch.payloads[0]
    snapshot = fixture.snapshots[maker_payload.market_id]
    target_price = _round_price(snapshot.mid_price(maker_payload.direction))
    return abs(maker_payload.price - target_price) > 1e-9


def _d3_clamp_active(batch: MevExecutionBatch) -> bool:
    return any(
        abs(payload.price - float(payload.metadata["level_target"])) > 1e-9
        for payload in batch.payloads
    )


def _build_pretty_summary(fixture, batch: MevExecutionBatch) -> str:
    clamp_state = "Inactive"

    if batch.playbook == "shadow_sweep":
        taker_payload, maker_payload = batch.payloads
        clamp_state = "Active" if _shadow_clamp_active(fixture, batch) else "Inactive"
        return (
            f"[{batch.route_id}] Target: {taker_payload.market_id} | "
            f"Action: {_format_quantity(taker_payload.size)} Taker {taker_payload.direction} @ {taker_payload.price:.2f}"
            f" -> {_format_quantity(maker_payload.size)} Maker {maker_payload.direction} @ {maker_payload.price:.2f} | "
            f"Clamp: {clamp_state}"
        )

    if batch.playbook == "mm_trap":
        maker_payload, taker_payload = batch.payloads
        clamp_state = "Active" if _mm_trap_clamp_active(fixture, batch) else "Inactive"
        return (
            f"[{batch.route_id}] Target: {taker_payload.market_id} | "
            f"Action: {_format_quantity(taker_payload.size)} Taker Vol -> Correlated Maker Notional: "
            f"{_format_currency(float(maker_payload.metadata['attack_notional']))} | "
            f"Clamp: {clamp_state}"
        )

    total_size = sum(payload.size for payload in batch.payloads)
    limit_price = max(float(payload.metadata["limit_price"]) for payload in batch.payloads)
    clamp_state = "Active" if _d3_clamp_active(batch) else "Inactive"
    return (
        f"[{batch.route_id}] Target: {batch.payloads[0].market_id} | "
        f"Action: {len(batch.payloads)}-Level Maker Grid {_format_quantity(total_size)} {batch.payloads[0].direction} Shares <= {limit_price:.2f} | "
        f"Clamp: {clamp_state}"
    )


def _write_audit_file(output_file: str, serialized_batches: list[dict[str, Any]]) -> Path:
    output_path = Path(output_file)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serialized_batches, indent=2, sort_keys=False), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run audit harness for isolated MEV execution playbooks.",
    )
    parser.add_argument(
        "--scenario",
        default="ALL",
        choices=["ALL", *[fixture.name for fixture in SCENARIO_FIXTURES]],
        help="Run all MEV audit scenarios or a single named regime fixture.",
    )
    parser.add_argument(
        "--max-capital",
        type=float,
        default=None,
        help="Override max capital for shadow sweep and D3 panic absorption scenarios.",
    )
    parser.add_argument(
        "--premium-pct",
        type=float,
        default=DEFAULT_SHADOW_PREMIUM_PCT,
        help="Override the shadow sweep passive premium percentage.",
    )
    parser.add_argument(
        "--attack-volume",
        type=float,
        default=DEFAULT_ATTACK_VOLUME,
        help="Override the MM predation taker ping volume.",
    )
    parser.add_argument(
        "--panic-limit",
        type=float,
        default=DEFAULT_D3_PANIC_LIMIT,
        help="Override the D3 panic absorption limit price.",
    )
    parser.add_argument(
        "--pretty-summary",
        action="store_true",
        help="Print a one-line operator summary above each serialized execution batch.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Write the serialized execution batch array to a JSON audit file.",
    )
    return parser


def _selected_fixtures(scenario_name: str):
    if scenario_name == "ALL":
        return SCENARIO_FIXTURES
    return tuple(fixture for fixture in SCENARIO_FIXTURES if fixture.name == scenario_name)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    shadow_max_capital = (
        float(args.max_capital)
        if args.max_capital is not None
        else DEFAULT_SHADOW_MAX_CAPITAL
    )
    d3_max_capital = (
        float(args.max_capital)
        if args.max_capital is not None
        else DEFAULT_D3_MAX_CAPITAL
    )

    fixtures = _selected_fixtures(args.scenario)
    if not fixtures:
        raise ValueError(f"Unknown scenario selection: {args.scenario}")

    combined_snapshots = {}
    for fixture in fixtures:
        combined_snapshots.update(fixture.snapshots)

    router = MevExecutionRouter(lambda market_id: combined_snapshots[market_id])
    dispatcher = MevDispatcher(router)
    serialized_batches: list[dict[str, Any]] = []

    for fixture in fixtures:
        if fixture.kind == "shadow_sweep":
            batch = dispatcher.on_mempool_whale_detected(
                build_shadow_signal(
                    max_capital=shadow_max_capital,
                    premium_pct=float(args.premium_pct),
                )
            )
        elif fixture.kind == "mm_trap":
            batch = dispatcher.on_mm_vulnerability_detected(
                build_mm_signal(attack_volume=float(args.attack_volume))
            )
        else:
            batch = dispatcher.on_uma_dispute_panic(
                build_dispute_signal(
                    max_capital=d3_max_capital,
                    panic_limit=float(args.panic_limit),
                )
            )

        serialized_batch = _serialize_batch_payload(batch)
        serialized_batches.append(serialized_batch)

        print(f"===== {fixture.title} :: {fixture.name} =====")
        print(fixture.description)
        if args.pretty_summary:
            print(_build_pretty_summary(fixture, batch))
        print(json.dumps(serialized_batch, indent=2, sort_keys=False))
        print()

    if args.output_file:
        output_path = _write_audit_file(args.output_file, serialized_batches)
        print(f"Wrote {len(serialized_batches)} batch record(s) to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())