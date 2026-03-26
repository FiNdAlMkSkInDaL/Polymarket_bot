from __future__ import annotations

from dataclasses import dataclass

from src.signals.mm_tracker import MarketMakerTracker


def _normalize_direction(direction: str) -> str:
    value = str(direction or "").strip().upper()
    if value not in {"YES", "NO"}:
        raise ValueError(f"Unsupported trap direction: {direction!r}")
    return value


def _opposite_direction(direction: str) -> str:
    normalized = _normalize_direction(direction)
    return "NO" if normalized == "YES" else "YES"


@dataclass(frozen=True, slots=True)
class CorrelatedOrderBookState:
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    yes_liquidity: float
    no_liquidity: float

    def best_bid(self, direction: str) -> float:
        normalized = _normalize_direction(direction)
        return float(self.yes_bid if normalized == "YES" else self.no_bid)

    def best_ask(self, direction: str) -> float:
        normalized = _normalize_direction(direction)
        return float(self.yes_ask if normalized == "YES" else self.no_ask)

    def spread(self, direction: str) -> float:
        bid = self.best_bid(direction)
        ask = self.best_ask(direction)
        if bid <= 0.0 or ask <= 0.0:
            return float("inf")
        return ask - bid

    def available_liquidity(self, direction: str) -> float:
        normalized = _normalize_direction(direction)
        return float(self.yes_liquidity if normalized == "YES" else self.no_liquidity)


@dataclass(frozen=True, slots=True)
class MMPredationSignal:
    target_market_id: str
    correlated_market_id: str
    maker_address: str
    v_attack: float
    trap_direction: str
    hedge_direction: str
    estimated_kappa: float
    correlated_spread: float
    correlated_available_liquidity: float
    signal_source: str = "SI-MM-Predation"


class MMPredationDetector:
    """Orchestrates fingerprint updates and emits isolated MM predation signals."""

    __slots__ = (
        "tracker",
        "target_spread_delta",
        "max_capital",
        "min_correlated_liquidity",
        "max_correlated_spread",
        "_correlated_markets",
        "_order_books",
    )

    def __init__(
        self,
        *,
        tracker: MarketMakerTracker | None = None,
        target_spread_delta: float = 0.01,
        max_capital: float = 100.0,
        min_correlated_liquidity: float = 25.0,
        max_correlated_spread: float = 0.03,
    ) -> None:
        self.tracker = tracker or MarketMakerTracker()
        self.target_spread_delta = float(target_spread_delta)
        self.max_capital = float(max_capital)
        self.min_correlated_liquidity = float(min_correlated_liquidity)
        self.max_correlated_spread = float(max_correlated_spread)
        self._correlated_markets: dict[str, set[str]] = {}
        self._order_books: dict[str, CorrelatedOrderBookState] = {}

    def register_correlation(self, target_market_id: str, correlated_market_id: str) -> None:
        target_key = str(target_market_id).strip()
        correlated_key = str(correlated_market_id).strip()
        self._correlated_markets.setdefault(target_key, set()).add(correlated_key)

    def set_order_book(self, market_id: str, order_book: CorrelatedOrderBookState) -> None:
        self._order_books[str(market_id).strip()] = order_book

    def evaluate_market_tick(
        self,
        target_market_id: str,
        maker_address: str,
        price_delta: float,
        taker_volume: float,
        correlated_market_id: str,
    ) -> MMPredationSignal | None:
        target_key = str(target_market_id).strip()
        correlated_key = str(correlated_market_id).strip()
        self.register_correlation(target_key, correlated_key)
        self.tracker.process_fill_event(maker_address, price_delta, taker_volume)

        vulnerable = dict(
            self.tracker.get_vulnerable_makers(
                target_spread_delta=self.target_spread_delta,
                max_capital=self.max_capital,
            )
        )
        required_attack_volume = vulnerable.get(str(maker_address).strip())
        if required_attack_volume is None:
            return None

        if correlated_key not in self._correlated_markets.get(target_key, set()):
            return None

        correlated_book = self._order_books.get(correlated_key)
        if correlated_book is None:
            return None

        trap_direction = "YES" if float(price_delta) >= 0.0 else "NO"
        hedge_direction = _opposite_direction(trap_direction)
        correlated_spread = correlated_book.spread(hedge_direction)
        correlated_liquidity = correlated_book.available_liquidity(hedge_direction)
        if correlated_spread > self.max_correlated_spread:
            return None
        if correlated_liquidity < max(self.min_correlated_liquidity, required_attack_volume):
            return None

        fingerprint = self.tracker.fingerprints[str(maker_address).strip()]
        return MMPredationSignal(
            target_market_id=target_key,
            correlated_market_id=correlated_key,
            maker_address=str(maker_address).strip(),
            v_attack=required_attack_volume,
            trap_direction=trap_direction,
            hedge_direction=hedge_direction,
            estimated_kappa=fingerprint.kappa_ewma,
            correlated_spread=correlated_spread,
            correlated_available_liquidity=correlated_liquidity,
        )