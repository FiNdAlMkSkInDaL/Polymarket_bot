#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.backtest.data_recorder import MarketDataRecorder
from src.backtest.strategy import _ReplayOrderBook

try:
    from src.backtest.strategy import OFI_REPLAY_TIME_STOP_VACUUM_RATIO
except ImportError:
    OFI_REPLAY_TIME_STOP_VACUUM_RATIO = 0.35

try:
    from src.backtest.strategy import OFI_REPLAY_TIME_STOP_SPREAD_MULTIPLIER
except ImportError:
    try:
        from src.backtest.strategy import OFI_REPLAY_SPREAD_EXPANSION_MULTIPLIER as OFI_REPLAY_TIME_STOP_SPREAD_MULTIPLIER
    except ImportError:
        OFI_REPLAY_TIME_STOP_SPREAD_MULTIPLIER = 1.75
from src.backtest.wfo_optimizer import _build_data_loader, _load_market_configs, generate_folds
from src.core.config import StrategyParams
from src.data.ohlcv import OHLCVAggregator
from src.data.websocket_client import TradeEvent
from src.signals.signal_framework import MetaStrategyController

try:
    from src.signals.ofi_momentum import OFIMomentumDetector
except ImportError:
    class OFIMomentumDetector:
        def __init__(
            self,
            market_id: str,
            *,
            no_asset_id: str = "",
            window_ms: int = 2000,
            threshold: float = 0.85,
            tvi_kappa: float = 1.0,
        ) -> None:
            if window_ms <= 0:
                raise ValueError("window_ms must be positive")
            if threshold <= 0 or threshold >= 1:
                raise ValueError("threshold must be between 0 and 1")
            if tvi_kappa < 0:
                raise ValueError("tvi_kappa must be non-negative")

            self.market_id = market_id
            self.no_asset_id = no_asset_id
            self.window_ms = int(window_ms)
            self.threshold = float(threshold)
            self.tvi_kappa = float(tvi_kappa)
            self._vi_window: deque[tuple[int, float]] = deque()
            self._rolling_sum = 0.0

        def record_top_of_book(
            self,
            bid_size: float,
            ask_size: float,
            *,
            timestamp_ms: int,
        ) -> float:
            vi = self._compute_vi(bid_size, ask_size)
            if vi is None:
                return self.rolling_vi

            self._prune(timestamp_ms)
            self._vi_window.append((timestamp_ms, vi))
            self._rolling_sum += vi
            self._prune(timestamp_ms)
            return self.rolling_vi

        @property
        def rolling_vi(self) -> float:
            if not self._vi_window:
                return 0.0
            return self._rolling_sum / len(self._vi_window)

        def _trade_verified_signal(
            self,
            rolling_vi: float,
            *,
            trade_aggregator: Any,
            timestamp_ms: int,
        ) -> tuple[float, float, float]:
            if trade_aggregator is None:
                return 0.0, 1.0, rolling_vi

            moments_fn = getattr(trade_aggregator, "trade_flow_moments", None)
            if callable(moments_fn):
                moments = moments_fn(self.window_ms, current_time_ms=timestamp_ms)
                if isinstance(moments, tuple) and len(moments) == 2:
                    buy_volume, sell_volume = moments
                    if float(buy_volume or 0.0) + float(sell_volume or 0.0) <= 0:
                        return 0.0, 1.0, rolling_vi

            imbalance_fn = getattr(trade_aggregator, "trade_flow_imbalance", None)
            if not callable(imbalance_fn):
                return 0.0, 1.0, rolling_vi

            trade_flow_imbalance = float(
                imbalance_fn(self.window_ms, current_time_ms=timestamp_ms) or 0.0
            )
            penalty = max(0.0, 1.0 - self.tvi_kappa * abs(rolling_vi - trade_flow_imbalance))
            return trade_flow_imbalance, penalty, rolling_vi * penalty

        def _prune(self, now_ms: int) -> None:
            cutoff = now_ms - self.window_ms
            while self._vi_window and self._vi_window[0][0] < cutoff:
                _, expired_vi = self._vi_window.popleft()
                self._rolling_sum -= expired_vi

        @staticmethod
        def _compute_vi(bid_size: float, ask_size: float) -> float | None:
            bid = float(bid_size)
            ask = float(ask_size)
            total = bid + ask
            if bid < 0 or ask < 0 or total <= 0:
                return None
            return (bid - ask) / total


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _decimal_min(left: Decimal, right: Decimal) -> Decimal:
    return left if left <= right else right


@dataclass(frozen=True)
class TraceParams:
    fold_index: int
    train_days: int
    test_days: int
    step_days: int
    embargo_days: int
    max_markets: int
    ofi_threshold: Decimal
    window_ms: int
    ofi_tvi_kappa: Decimal
    ofi_toxicity_scale_threshold: Decimal
    ofi_toxicity_size_boost_max: Decimal
    take_profit_pct: Decimal
    stop_loss_pct: Decimal
    kelly_fraction: Decimal
    max_trade_size_usd: Decimal
    initial_cash: Decimal
    signal_cooldown_minutes: Decimal


@dataclass(frozen=True)
class SignalDropFunnel:
    total_book_updates: int
    total_ofi_threshold_breaches: int
    toxicity_suppressions: int
    depth_vacuum_suppressions: int
    final_valid_signals: int
    tvi_penalty_suppressions: int
    cooldown_suppressions: int
    meta_vetoes: int
    size_floor_suppressions: int


@dataclass(frozen=True)
class TraceWindow:
    fold_index: int
    date_start: str
    date_end: str
    event_start_iso: str | None
    event_end_iso: str | None


@dataclass(frozen=True)
class TraceReport:
    window: TraceWindow
    params: TraceParams
    target_market_count: int
    target_no_asset_ids: tuple[str, ...]
    funnel: SignalDropFunnel


class _MarketRuntime:
    def __init__(self, market_id: str, no_asset_id: str, params: TraceParams) -> None:
        self.market_id = market_id
        self.no_asset_id = no_asset_id
        self.book = _ReplayOrderBook(no_asset_id)
        self.agg = OHLCVAggregator(no_asset_id)
        self.detector = OFIMomentumDetector(
            market_id=market_id,
            no_asset_id=no_asset_id,
            window_ms=params.window_ms,
            threshold=float(params.ofi_threshold),
            tvi_kappa=float(params.ofi_tvi_kappa),
        )
        self.last_signal_time_ms = 0


def _iso(timestamp_s: float | None) -> str | None:
    if timestamp_s is None:
        return None
    return datetime.fromtimestamp(timestamp_s, tz=timezone.utc).isoformat()


def _toxicity_multiplier(toxicity_index: Decimal, params: TraceParams) -> Decimal:
    threshold = params.ofi_toxicity_scale_threshold
    ceiling = params.ofi_toxicity_size_boost_max
    if toxicity_index <= threshold or ceiling <= Decimal("1"):
        return Decimal("1")
    progress = (toxicity_index - threshold) / max(Decimal("0.000001"), Decimal("1") - threshold)
    progress = min(Decimal("1"), max(Decimal("0"), progress))
    return Decimal("1") + progress * (ceiling - Decimal("1"))


def _depth_vacuum_triggered(book: _ReplayOrderBook, best_bid: Decimal, best_ask: Decimal, params: TraceParams) -> bool:
    bid_depth, ask_depth = book.top_depths_usd()
    bid_baseline = book.top_depth_ewma("bid")
    ask_baseline = book.top_depth_ewma("ask")
    spread = max(Decimal("0"), best_ask - best_bid)
    baseline_spread = max(Decimal("0.01"), best_ask * params.stop_loss_pct)

    bid_depth_dec = _decimal(bid_depth)
    ask_depth_dec = _decimal(ask_depth)
    bid_baseline_dec = _decimal(bid_baseline)
    ask_baseline_dec = _decimal(ask_baseline)
    vacuum_ratio = _decimal(OFI_REPLAY_TIME_STOP_VACUUM_RATIO)
    spread_multiplier = _decimal(OFI_REPLAY_TIME_STOP_SPREAD_MULTIPLIER)

    bid_vacuum = bid_depth_dec > 0 and bid_baseline_dec > 0 and bid_depth_dec < bid_baseline_dec * vacuum_ratio
    ask_vacuum = ask_depth_dec > 0 and ask_baseline_dec > 0 and ask_depth_dec < ask_baseline_dec * vacuum_ratio
    spread_blown_out = spread >= baseline_spread * spread_multiplier
    return (bid_vacuum or ask_vacuum) and spread_blown_out


def _trade_event_from_market_event(event: Any) -> TradeEvent | None:
    data = event.data
    try:
        price = float(data.get("price", 0.0) or 0.0)
        size = float(data.get("size", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if price <= 0.0 or size <= 0.0:
        return None
    return TradeEvent(
        timestamp=float(event.timestamp),
        market_id=str(data.get("market_id", data.get("market", "")) or ""),
        asset_id=str(event.asset_id),
        side=str(data.get("side", "buy") or "buy").lower(),
        price=price,
        size=size,
        is_yes=False,
        is_taker=bool(data.get("is_taker", False)),
    )


def _load_trace_market_configs(data_dir: str, market_configs_path: str) -> list[dict[str, str]]:
    market_configs_file = Path(market_configs_path)
    if not market_configs_file.is_absolute():
        market_configs_file = Path.cwd() / market_configs_file
    if market_configs_file.exists():
        raw = json.loads(market_configs_file.read_text(encoding="utf-8"))
        configs: list[dict[str, str]] = []
        for entry in raw:
            market_id = str(entry.get("market_id", "") or "")
            yes_asset_id = str(entry.get("yes_asset_id", entry.get("yes_id", "")) or "")
            no_asset_id = str(entry.get("no_asset_id", entry.get("no_id", "")) or "")
            if market_id and yes_asset_id and no_asset_id:
                configs.append(
                    {
                        "market_id": market_id,
                        "yes_asset_id": yes_asset_id,
                        "no_asset_id": no_asset_id,
                    }
                )
        if configs:
            return configs

    try:
        return _load_market_configs(data_dir, market_configs_path)
    except TypeError:
        return _load_market_configs(data_dir)


def _resolve_trace_data_dir(data_dir: str) -> str:
    path = Path(data_dir)
    if path.name in {"raw_ticks", "ticks"}:
        return str(path.parent)
    return str(path)


def trace_ofi_fold(
    *,
    data_dir: str,
    market_configs_path: str,
    fold_index: int,
    train_days: int,
    test_days: int,
    step_days: int,
    embargo_days: int,
    max_markets: int,
    ofi_threshold: Decimal,
    window_ms: int,
    ofi_tvi_kappa: Decimal,
    ofi_toxicity_scale_threshold: Decimal,
    ofi_toxicity_size_boost_max: Decimal,
    take_profit_pct: Decimal,
    stop_loss_pct: Decimal,
    kelly_fraction: Decimal,
    max_trade_size_usd: Decimal,
    initial_cash: Decimal,
    signal_cooldown_minutes: Decimal,
) -> TraceReport:
    trace_data_dir = _resolve_trace_data_dir(data_dir)
    available_dates = MarketDataRecorder.available_dates(trace_data_dir)
    folds = generate_folds(
        available_dates,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        embargo_days=embargo_days,
    )
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"fold_index {fold_index} out of range for {len(folds)} folds")

    market_configs = _load_trace_market_configs(trace_data_dir, market_configs_path)
    if not market_configs:
        raise FileNotFoundError(f"no market configs found at {market_configs_path}")
    selected_configs = market_configs[:max_markets]
    no_asset_ids = tuple(config["no_asset_id"] for config in selected_configs)

    fold = folds[fold_index]
    loader = _build_data_loader(trace_data_dir, fold.test_dates, asset_ids=set(no_asset_ids))
    if loader is None:
        loader = _build_data_loader(trace_data_dir, fold.test_dates)
    if loader is None:
        raise FileNotFoundError("no replay data found for selected fold")

    params = TraceParams(
        fold_index=fold_index,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        embargo_days=embargo_days,
        max_markets=max_markets,
        ofi_threshold=ofi_threshold,
        window_ms=window_ms,
        ofi_tvi_kappa=ofi_tvi_kappa,
        ofi_toxicity_scale_threshold=ofi_toxicity_scale_threshold,
        ofi_toxicity_size_boost_max=ofi_toxicity_size_boost_max,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        kelly_fraction=kelly_fraction,
        max_trade_size_usd=max_trade_size_usd,
        initial_cash=initial_cash,
        signal_cooldown_minutes=signal_cooldown_minutes,
    )
    states = {
        config["no_asset_id"]: _MarketRuntime(config["market_id"], config["no_asset_id"], params)
        for config in selected_configs
    }
    meta_controller = MetaStrategyController()
    cooldown_ms = int((signal_cooldown_minutes * Decimal("60") * Decimal("1000")).to_integral_value())
    trade_budget = _decimal_min(initial_cash * kelly_fraction, max_trade_size_usd)

    total_book_updates = 0
    total_ofi_threshold_breaches = 0
    toxicity_suppressions = 0
    depth_vacuum_suppressions = 0
    final_valid_signals = 0
    tvi_penalty_suppressions = 0
    cooldown_suppressions = 0
    meta_vetoes = 0
    size_floor_suppressions = 0
    first_event_ts: float | None = None
    last_event_ts: float | None = None
    total_l2_events_seen = 0

    for event in loader:
        if event.event_type in ("l2_delta", "l2_snapshot"):
            total_l2_events_seen += 1

        state = states.get(event.asset_id)
        if state is None:
            continue
        if first_event_ts is None:
            first_event_ts = float(event.timestamp)
        last_event_ts = float(event.timestamp)

        if event.event_type == "trade":
            trade_event = _trade_event_from_market_event(event)
            if trade_event is not None:
                state.agg.on_trade(trade_event)
            continue

        if event.event_type not in ("l2_delta", "l2_snapshot"):
            continue

        state.book.apply_event(event.data, float(event.timestamp))
        total_book_updates += 1

        bid_levels = state.book.levels("bid", 1)
        ask_levels = state.book.levels("ask", 1)
        if not bid_levels or not ask_levels:
            continue

        timestamp_ms = int(_decimal(event.timestamp) * Decimal("1000"))
        raw_rolling_vi = state.detector.record_top_of_book(
            float(bid_levels[0].size),
            float(ask_levels[0].size),
            timestamp_ms=timestamp_ms,
        )
        trade_flow_imbalance, _tvi_multiplier, adjusted_vi = state.detector._trade_verified_signal(
            raw_rolling_vi,
            trade_aggregator=state.agg,
            timestamp_ms=timestamp_ms,
        )

        raw_vi_dec = _decimal(raw_rolling_vi)
        adjusted_vi_dec = _decimal(adjusted_vi)
        best_bid_dec = _decimal(state.book.best_bid)
        best_ask_dec = _decimal(state.book.best_ask)
        toxicity_metrics = state.book.toxicity_metrics("BUY")
        toxicity_index_dec = _decimal(toxicity_metrics.get("toxicity_index", 0.0))
        del trade_flow_imbalance

        if raw_vi_dec <= ofi_threshold:
            continue
        total_ofi_threshold_breaches += 1

        if toxicity_index_dec >= ofi_toxicity_scale_threshold:
            toxicity_suppressions += 1
        if _depth_vacuum_triggered(state.book, best_bid_dec, best_ask_dec, params):
            depth_vacuum_suppressions += 1
        if adjusted_vi_dec <= ofi_threshold:
            tvi_penalty_suppressions += 1
            continue

        meta_decision = meta_controller.evaluate("ofi_momentum", 0.5)
        if meta_decision.vetoed:
            meta_vetoes += 1
            continue
        if state.last_signal_time_ms > 0 and timestamp_ms - state.last_signal_time_ms < cooldown_ms:
            cooldown_suppressions += 1
            continue
        if best_ask_dec <= 0:
            continue

        size_multiplier = _toxicity_multiplier(toxicity_index_dec, params)
        size = (trade_budget / best_ask_dec) * size_multiplier
        if size < Decimal("1"):
            size_floor_suppressions += 1
            continue

        final_valid_signals += 1
        state.last_signal_time_ms = timestamp_ms

    if total_l2_events_seen == 0:
        window_start = fold.test_dates[0] if fold.test_dates else "unknown"
        window_end = fold.test_dates[-1] if fold.test_dates else "unknown"
        raise RuntimeError(
            "fatal: selected date window contains 0 L2 events; "
            f"data_dir={data_dir} resolved_data_dir={trace_data_dir} "
            f"window={window_start}..{window_end}"
        )

    return TraceReport(
        window=TraceWindow(
            fold_index=fold.index,
            date_start=fold.test_dates[0],
            date_end=fold.test_dates[-1],
            event_start_iso=_iso(first_event_ts),
            event_end_iso=_iso(last_event_ts),
        ),
        params=params,
        target_market_count=len(selected_configs),
        target_no_asset_ids=no_asset_ids,
        funnel=SignalDropFunnel(
            total_book_updates=total_book_updates,
            total_ofi_threshold_breaches=total_ofi_threshold_breaches,
            toxicity_suppressions=toxicity_suppressions,
            depth_vacuum_suppressions=depth_vacuum_suppressions,
            final_valid_signals=final_valid_signals,
            tvi_penalty_suppressions=tvi_penalty_suppressions,
            cooldown_suppressions=cooldown_suppressions,
            meta_vetoes=meta_vetoes,
            size_floor_suppressions=size_floor_suppressions,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    defaults = StrategyParams()
    parser = argparse.ArgumentParser(description="Trace OFI signal drop-funnel for a WFO fold")
    parser.add_argument("--data-dir", default="data/vps_march2026")
    parser.add_argument("--market-configs", default="data/market_map_top25.json")
    parser.add_argument("--fold-index", type=int, default=2)
    parser.add_argument("--train-days", type=int, default=35)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--max-markets", type=int, default=25)
    parser.add_argument("--ofi-threshold", default=str(getattr(defaults, "ofi_threshold", 0.75)))
    parser.add_argument("--window-ms", type=int, default=int(getattr(defaults, "window_ms", 2000)))
    parser.add_argument("--ofi-tvi-kappa", default=str(getattr(defaults, "ofi_tvi_kappa", 1.0)))
    parser.add_argument(
        "--ofi-toxicity-scale-threshold",
        default=str(getattr(defaults, "ofi_toxicity_scale_threshold", 0.60)),
    )
    parser.add_argument(
        "--ofi-toxicity-size-boost-max",
        default=str(getattr(defaults, "ofi_toxicity_size_boost_max", 2.0)),
    )
    parser.add_argument("--take-profit-pct", default=str(getattr(defaults, "take_profit_pct", 0.02)))
    parser.add_argument("--stop-loss-pct", default=str(getattr(defaults, "stop_loss_pct", 0.02)))
    parser.add_argument("--kelly-fraction", default=str(getattr(defaults, "kelly_fraction", 0.25)))
    parser.add_argument("--max-trade-size-usd", default=str(getattr(defaults, "max_trade_size_usd", 15.0)))
    parser.add_argument("--initial-cash", default="1000")
    parser.add_argument(
        "--signal-cooldown-minutes",
        default=str(getattr(defaults, "signal_cooldown_minutes", 5.0)),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = trace_ofi_fold(
        data_dir=args.data_dir,
        market_configs_path=args.market_configs,
        fold_index=args.fold_index,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        embargo_days=args.embargo_days,
        max_markets=args.max_markets,
        ofi_threshold=_decimal(args.ofi_threshold),
        window_ms=args.window_ms,
        ofi_tvi_kappa=_decimal(args.ofi_tvi_kappa),
        ofi_toxicity_scale_threshold=_decimal(args.ofi_toxicity_scale_threshold),
        ofi_toxicity_size_boost_max=_decimal(args.ofi_toxicity_size_boost_max),
        take_profit_pct=_decimal(args.take_profit_pct),
        stop_loss_pct=_decimal(args.stop_loss_pct),
        kelly_fraction=_decimal(args.kelly_fraction),
        max_trade_size_usd=_decimal(args.max_trade_size_usd),
        initial_cash=_decimal(args.initial_cash),
        signal_cooldown_minutes=_decimal(args.signal_cooldown_minutes),
    )
    print("OFI Signal Drop Funnel")
    print(
        f"{report.funnel.total_book_updates} -> {report.funnel.total_ofi_threshold_breaches} -> "
        f"{report.funnel.toxicity_suppressions} -> {report.funnel.depth_vacuum_suppressions} -> "
        f"{report.funnel.final_valid_signals}"
    )
    print(json.dumps(asdict(report), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())