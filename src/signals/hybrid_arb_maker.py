from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from importlib import import_module
from typing import Any, Callable, Literal, Protocol, Sequence

from src.execution.priority_context import PriorityOrderContext
from src.models.amm_pricing import (
    DEFAULT_GAS_AND_FEE_BUFFER_CENTS,
    ArbitrageSpread,
    compute_delta_1,
    compute_delta_2,
    quote_binary_cpmm_trade,
    quote_binary_lmsr_trade,
)
from src.models.arb_risk_manager import ArbSizingResult, calculate_safe_arb_size
from src.signals.base_strategy import BaseStrategy
from src.signals.obi_scalper import ObiScalper


_ONE = Decimal("1")

AmmModel = Literal["CPMM", "LMSR"]


class ClockMs(Protocol):
    def __call__(self) -> int:
        ...


@dataclass(slots=True)
class _MarketState:
    best_bid: Decimal | None = None
    best_bid_size: Decimal | None = None
    best_ask: Decimal | None = None
    best_ask_size: Decimal | None = None
    last_delta_1: ArbitrageSpread | None = None
    last_delta_2: ArbitrageSpread | None = None
    last_reserve_snapshot: dict[str, Decimal] | None = None


@dataclass(slots=True)
class HybridArbMaker(BaseStrategy):
    dispatcher: Any | None = None
    market_catalog: Any | None = None
    clock: ClockMs = field(default=lambda: int(time.time() * 1000))
    amm_model: AmmModel = "CPMM"
    order_size_shares: Decimal = Decimal("10")
    capital_cap_usd: Decimal = Decimal("5000")
    max_trade_size_usd: Decimal = Decimal("50")
    min_trade_notional_usd: Decimal = Decimal("10")
    gas_and_fee_buffer_cents: Decimal = DEFAULT_GAS_AND_FEE_BUFFER_CENTS
    cooldown_ms: int = 1_000
    signal_source: str = "MANUAL"
    reserve_provider: Callable[[str], Any] | None = None
    _market_state: dict[str, _MarketState] = field(init=False, repr=False)
    _last_signal_at_ms: dict[tuple[str, str], int] = field(init=False, repr=False)
    _reserve_provider: Callable[[str], Any] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        BaseStrategy.__init__(
            self,
            dispatcher=self.dispatcher,
            market_catalog=self.market_catalog,
            clock=self.clock,
        )
        if not callable(self.clock):
            raise ValueError("clock must be callable")

        self.amm_model = str(self.amm_model or "").strip().upper()  # type: ignore[assignment]
        if self.amm_model not in {"CPMM", "LMSR"}:
            raise ValueError("amm_model must be 'CPMM' or 'LMSR'")

        self.order_size_shares = self._as_decimal(self.order_size_shares, name="order_size_shares")
        self.capital_cap_usd = self._as_decimal(self.capital_cap_usd, name="capital_cap_usd")
        self.max_trade_size_usd = self._as_decimal(self.max_trade_size_usd, name="max_trade_size_usd")
        self.min_trade_notional_usd = self._as_decimal(self.min_trade_notional_usd, name="min_trade_notional_usd")
        self.gas_and_fee_buffer_cents = self._as_decimal(
            self.gas_and_fee_buffer_cents,
            name="gas_and_fee_buffer_cents",
        )
        if self.order_size_shares <= Decimal("0"):
            raise ValueError("order_size_shares must be strictly positive")
        if self.capital_cap_usd <= Decimal("0"):
            raise ValueError("capital_cap_usd must be strictly positive")
        if self.max_trade_size_usd <= Decimal("0"):
            raise ValueError("max_trade_size_usd must be strictly positive")
        if self.min_trade_notional_usd <= Decimal("0"):
            raise ValueError("min_trade_notional_usd must be strictly positive")
        if self.gas_and_fee_buffer_cents <= Decimal("0"):
            raise ValueError("gas_and_fee_buffer_cents must be strictly positive")
        if not isinstance(self.cooldown_ms, int) or self.cooldown_ms < 0:
            raise ValueError("cooldown_ms must be a non-negative int")

        self.signal_source = str(self.signal_source or "").strip().upper() or "MANUAL"
        if self.signal_source not in {"OFI", "SI9", "SI10", "CONTAGION", "MANUAL", "CTF", "REWARD"}:
            raise ValueError(f"Unsupported signal_source: {self.signal_source!r}")
        if self.reserve_provider is not None and not callable(self.reserve_provider):
            raise ValueError("reserve_provider must be callable")

        self._market_state = {}
        self._last_signal_at_ms = {}
        self._reserve_provider = self.reserve_provider

    def bind_dispatcher(self, dispatcher: Any) -> None:
        self.dispatcher = dispatcher
        super().bind_dispatcher(dispatcher)

    def bind_market_catalog(self, market_catalog: Any) -> None:
        self.market_catalog = market_catalog
        super().bind_market_catalog(market_catalog)

    def bind_clock(self, clock) -> None:
        self.clock = clock
        super().bind_clock(clock)

    def bind_reserve_provider(self, reserve_provider: Callable[[str], Any] | None) -> None:
        if reserve_provider is not None and not callable(reserve_provider):
            raise ValueError("reserve_provider must be callable")
        self.reserve_provider = reserve_provider
        self._reserve_provider = reserve_provider

    def on_bbo_update(
        self,
        market_id: str,
        top_bids: Sequence[dict[str, Any]],
        top_asks: Sequence[dict[str, Any]],
    ) -> None:
        normalized_market_id = str(market_id or "").strip()
        if not normalized_market_id:
            return

        bids = ObiScalper._normalize_levels(top_bids)
        asks = ObiScalper._normalize_levels(top_asks)
        if not bids or not asks:
            return

        state = self._state_for(normalized_market_id)
        state.best_bid = bids[0].price
        state.best_bid_size = bids[0].size
        state.best_ask = asks[0].price
        state.best_ask_size = asks[0].size

    def on_trade(self, market_id: str, trade_data: dict[str, Any]) -> None:
        return None

    def on_tick(self) -> None:
        for market_id, state in self._market_state.items():
            if state.best_bid is None or state.best_ask is None:
                continue

            reserve_snapshot = self._get_pool_reserves(market_id)
            if reserve_snapshot is None:
                continue
            normalized_reserves = self._coerce_mapping(reserve_snapshot)
            state.last_reserve_snapshot = normalized_reserves

            amm_buy_price, amm_sell_price = self._quote_amm_prices(normalized_reserves)
            delta_1 = compute_delta_1(
                amm_sell_price=amm_sell_price,
                clob_best_ask=state.best_ask,
                order_size_shares=self.order_size_shares,
                gas_and_fee_buffer_cents=self.gas_and_fee_buffer_cents,
            )
            delta_2 = compute_delta_2(
                amm_buy_price=amm_buy_price,
                clob_best_bid=state.best_bid,
                order_size_shares=self.order_size_shares,
                gas_and_fee_buffer_cents=self.gas_and_fee_buffer_cents,
            )
            state.last_delta_1 = delta_1
            state.last_delta_2 = delta_2

            if delta_1.is_arbitrage_present and state.best_ask_size is not None:
                safe_size = self._calculate_safe_size(
                    clob_available_volume_at_bbo=state.best_ask_size,
                    clob_target_price=state.best_ask,
                    reserve_snapshot=normalized_reserves,
                )
                if safe_size.size_usd >= self.min_trade_notional_usd:
                    self._submit_if_outside_cooldown(
                        market_id=market_id,
                        direction="CLOB_TO_AMM",
                        context=self._build_context(
                            market_id=market_id,
                            execution_side="BUY",
                            clob_price=state.best_ask,
                            spread=delta_1,
                            safe_size=safe_size,
                            amm_buy_price=amm_buy_price,
                            amm_sell_price=amm_sell_price,
                        ),
                    )
            if delta_2.is_arbitrage_present and state.best_bid_size is not None:
                safe_size = self._calculate_safe_size(
                    clob_available_volume_at_bbo=state.best_bid_size,
                    clob_target_price=state.best_bid,
                    reserve_snapshot=normalized_reserves,
                )
                if safe_size.size_usd >= self.min_trade_notional_usd:
                    self._submit_if_outside_cooldown(
                        market_id=market_id,
                        direction="AMM_TO_CLOB",
                        context=self._build_context(
                            market_id=market_id,
                            execution_side="SELL",
                            clob_price=state.best_bid,
                            spread=delta_2,
                            safe_size=safe_size,
                            amm_buy_price=amm_buy_price,
                            amm_sell_price=amm_sell_price,
                        ),
                    )

    @property
    def latest_dislocations_by_market(self) -> dict[str, dict[str, ArbitrageSpread | None]]:
        return {
            market_id: {
                "delta_1": state.last_delta_1,
                "delta_2": state.last_delta_2,
            }
            for market_id, state in self._market_state.items()
        }

    def _submit_if_outside_cooldown(
        self,
        *,
        market_id: str,
        direction: str,
        context: PriorityOrderContext,
    ) -> None:
        dedup_key = (market_id, direction)
        last_signal_at_ms = self._last_signal_at_ms.get(dedup_key)
        current_timestamp_ms = self.current_timestamp_ms
        if last_signal_at_ms is not None and current_timestamp_ms - last_signal_at_ms < self.cooldown_ms:
            return
        self.submit_order(context)
        self._last_signal_at_ms[dedup_key] = current_timestamp_ms

    def _build_context(
        self,
        *,
        market_id: str,
        execution_side: Literal["BUY", "SELL"],
        clob_price: Decimal,
        spread: ArbitrageSpread,
        safe_size: ArbSizingResult,
        amm_buy_price: Decimal,
        amm_sell_price: Decimal,
    ) -> PriorityOrderContext:
        return PriorityOrderContext(
            market_id=market_id,
            side="YES",
            signal_source=self.signal_source,
            conviction_scalar=_ONE,
            target_price=clob_price,
            anchor_volume=safe_size.size_shares,
            max_capital=safe_size.size_usd,
            signal_metadata={
                "strategy": "hybrid_arb_maker",
                "liquidity_intent": "TAKER",
                "post_only": False,
                "time_in_force": "IOC",
                "execution_side": execution_side,
                "arb_direction": spread.direction,
                "quote_id": f"hybrid_arb_maker:{market_id}:{spread.direction.lower()}",
                "amm_model": self.amm_model,
                "amm_buy_price": str(amm_buy_price),
                "amm_sell_price": str(amm_sell_price),
                "gross_spread_cents": str(spread.gross_spread_cents),
                "net_spread_cents": str(spread.net_spread_cents),
                "gross_total_cents": str(spread.gross_total_cents),
                "net_total_cents": str(spread.net_total_cents),
                "gas_and_fee_buffer_cents": str(spread.gas_and_fee_buffer_cents),
                "safe_size_shares": str(safe_size.size_shares),
                "safe_size_usd": str(safe_size.size_usd),
                "size_cap_source": safe_size.capped_by,
                "min_trade_notional_usd": str(self.min_trade_notional_usd),
                "entry_theory": "hybrid_venue_dislocation_arb",
            },
        )

    def _quote_amm_prices(self, reserve_snapshot: Any) -> tuple[Decimal, Decimal]:
        snapshot = self._coerce_mapping(reserve_snapshot)
        if self.amm_model == "CPMM":
            amm_buy = quote_binary_cpmm_trade(
                yes_reserve=snapshot["yes_reserve"],
                no_reserve=snapshot["no_reserve"],
                outcome="YES",
                side="BUY",
                shares=self.order_size_shares,
            ).average_price
            amm_sell = quote_binary_cpmm_trade(
                yes_reserve=snapshot["yes_reserve"],
                no_reserve=snapshot["no_reserve"],
                outcome="YES",
                side="SELL",
                shares=self.order_size_shares,
            ).average_price
            return amm_buy, amm_sell

        amm_buy = quote_binary_lmsr_trade(
            yes_inventory=snapshot["yes_inventory"],
            no_inventory=snapshot["no_inventory"],
            liquidity=snapshot["liquidity"],
            outcome="YES",
            side="BUY",
            shares=self.order_size_shares,
        ).average_price
        amm_sell = quote_binary_lmsr_trade(
            yes_inventory=snapshot["yes_inventory"],
            no_inventory=snapshot["no_inventory"],
            liquidity=snapshot["liquidity"],
            outcome="YES",
            side="SELL",
            shares=self.order_size_shares,
        ).average_price
        return amm_buy, amm_sell

    def _get_pool_reserves(self, market_id: str) -> Any | None:
        provider = self._reserve_provider
        if provider is None:
            provider = self._load_default_reserve_provider()
            self._reserve_provider = provider
        try:
            return provider(market_id)
        except Exception:
            return None

    def _calculate_safe_size(
        self,
        *,
        clob_available_volume_at_bbo: Decimal,
        clob_target_price: Decimal,
        reserve_snapshot: dict[str, Decimal],
    ) -> ArbSizingResult:
        return calculate_safe_arb_size(
            clob_available_volume_at_bbo=clob_available_volume_at_bbo,
            amm_reserves=reserve_snapshot,
            capital_cap_usd=self.capital_cap_usd,
            max_trade_size_usd=self.max_trade_size_usd,
            clob_target_price=clob_target_price,
            amm_model=self.amm_model,
            outcome="YES",
        )

    @staticmethod
    def _load_default_reserve_provider() -> Callable[[str], Any]:
        module = import_module("src.data.alchemy_rpc_client")
        provider = getattr(module, "get_pool_reserves", None)
        if provider is None or not callable(provider):
            raise RuntimeError("src.data.alchemy_rpc_client.get_pool_reserves is required for HybridArbMaker")
        return provider

    def _state_for(self, market_id: str) -> _MarketState:
        state = self._market_state.get(market_id)
        if state is None:
            state = _MarketState()
            self._market_state[market_id] = state
        return state

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Decimal]:
        if not isinstance(value, dict):
            raise ValueError("Pool reserve snapshot must be a dict-like mapping")
        return {
            str(key): HybridArbMaker._as_decimal(raw_value, name=str(key))
            for key, raw_value in value.items()
        }

    @staticmethod
    def _as_decimal(value: Any, *, name: str) -> Decimal:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError(f"{name} must be finite")
        return decimal_value