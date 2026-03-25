from __future__ import annotations

import src.signals.combinatorial_arb as combinatorial_arb

from src.data.arb_clusters import ArbCluster
from src.data.market_discovery import MarketInfo
from src.data.orderbook import OrderbookTracker
from src.signals.combinatorial_arb import ComboArbDetector


def _market(
    condition_id: str,
    yes_token_id: str,
    no_token_id: str,
    question: str,
    *,
    liquidity_usd: float,
    daily_volume_usd: float = 1000.0,
) -> MarketInfo:
    return MarketInfo(
        condition_id=condition_id,
        question=question,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        daily_volume_usd=daily_volume_usd,
        end_date=None,
        active=True,
        event_id="EVENT-1",
        liquidity_usd=liquidity_usd,
        neg_risk=True,
    )


def _book(
    best_bid: float,
    best_ask: float,
    bid_size: float,
    ask_size: float,
    *,
    timestamp: float = 1.0,
) -> OrderbookTracker:
    book = OrderbookTracker("unused")
    book.on_book_snapshot(
        {
            "bids": [{"price": str(best_bid), "size": str(bid_size)}],
            "asks": [{"price": str(best_ask), "size": str(ask_size)}],
            "timestamp": timestamp,
        }
    )
    return book


class FakeAgg:
    def __init__(self, *, sigma: float, current_price: float):
        self.rolling_volatility_ewma = sigma
        self.current_price = current_price

    def trade_flow_imbalance(self, window_ms: int, *, current_time_ms: int | None = None) -> float:
        return 0.0


def test_combo_arb_signal_splits_maker_and_takers():

    books = {
        "YES_A": _book(0.30, 0.34, 400.0, 200.0),
        "YES_B": _book(0.28, 0.38, 500.0, 200.0),
        "YES_C": _book(0.29, 0.31, 450.0, 200.0),
    }
    detector = ComboArbDetector(books)
    cluster = ArbCluster(
        event_id="EVENT-1",
        legs=[
            _market("MKT_A", "YES_A", "NO_A", "Outcome A", liquidity_usd=800.0),
            _market("MKT_B", "YES_B", "NO_B", "Outcome B", liquidity_usd=600.0),
            _market("MKT_C", "YES_C", "NO_C", "Outcome C", liquidity_usd=900.0),
        ],
    )

    signal = detector.evaluate(cluster=cluster, wallet_balance=1000.0)

    assert signal is not None
    assert signal.maker_leg is not None
    assert signal.maker_leg.yes_token_id == "YES_B"
    assert signal.maker_leg.routing_reason == (
        "widest_spread spread_width=0.1000 next_spread_width=0.0400"
    )
    assert [leg.yes_token_id for leg in signal.taker_legs] == ["YES_A", "YES_C"]
    assert signal.maker_leg.target_price == 0.29
    assert [leg.target_price for leg in signal.taker_legs] == [0.34, 0.31]

    all_shares = [leg.target_shares for leg in signal.legs]
    assert all_shares == [signal.target_shares, signal.target_shares, signal.target_shares]
    assert round(signal.target_shares, 2) == 26.6
    assert signal.total_collateral == 25.0


def test_combo_arb_detector_prefers_thinnest_leg_when_spreads_tie():

    books = {
        "YES_A": _book(0.30, 0.35, 400.0, 200.0),
        "YES_B": _book(0.29, 0.34, 180.0, 200.0),
        "YES_C": _book(0.28, 0.33, 450.0, 200.0),
    }
    detector = ComboArbDetector(books)
    cluster = ArbCluster(
        event_id="EVENT-1",
        legs=[
            _market("MKT_A", "YES_A", "NO_A", "Outcome A", liquidity_usd=900.0),
            _market("MKT_B", "YES_B", "NO_B", "Outcome B", liquidity_usd=300.0),
            _market("MKT_C", "YES_C", "NO_C", "Outcome C", liquidity_usd=950.0),
        ],
    )

    signal = detector.evaluate_cluster(cluster, wallet_balance=1000.0)

    assert signal is not None
    assert signal.maker_leg is not None
    assert signal.maker_leg.yes_token_id == "YES_B"
    assert signal.maker_leg.routing_reason == (
        "thinnest_bid_depth bid_depth_usd=52.20 spread_width=0.0500"
    )
    assert signal.maker_leg.bid_depth_usd < min(leg.bid_depth_usd for leg in signal.taker_legs)
    assert all(leg.target_shares == signal.target_shares for leg in signal.taker_legs)


def test_combo_arb_detector_rejects_absurd_edge_on_dead_books():

    books = {
        "YES_A": _book(0.28, 0.32, 400.0, 200.0),
        "YES_B": _book(0.27, 0.31, 500.0, 200.0),
        "YES_C": _book(0.29, 0.33, 450.0, 200.0),
    }
    detector = ComboArbDetector(books)
    cluster = ArbCluster(
        event_id="EVENT-1",
        legs=[
            _market("MKT_A", "YES_A", "NO_A", "Outcome A", liquidity_usd=800.0),
            _market("MKT_B", "YES_B", "NO_B", "Outcome B", liquidity_usd=600.0),
            _market("MKT_C", "YES_C", "NO_C", "Outcome C", liquidity_usd=900.0),
        ],
    )

    signal = detector.evaluate_cluster(cluster, wallet_balance=1000.0)

    assert signal is None


def test_combo_arb_detector_allows_boundary_sum_bids_at_eighty_five_percent():

    books = {
        "YES_A": _book(0.28, 0.32, 400.0, 200.0),
        "YES_B": _book(0.28, 0.31, 500.0, 200.0),
        "YES_C": _book(0.29, 0.33, 450.0, 200.0),
    }
    detector = ComboArbDetector(books)
    cluster = ArbCluster(
        event_id="EVENT-1",
        legs=[
            _market("MKT_A", "YES_A", "NO_A", "Outcome A", liquidity_usd=800.0),
            _market("MKT_B", "YES_B", "NO_B", "Outcome B", liquidity_usd=600.0),
            _market("MKT_C", "YES_C", "NO_C", "Outcome C", liquidity_usd=900.0),
        ],
    )

    signal = detector.evaluate_cluster(cluster, wallet_balance=1000.0)

    assert signal is not None
    assert signal.sum_best_bids == 0.85
    assert signal.edge_cents == 13.0


def test_combo_arb_guardrail_logs_only_once_within_throttle_window(monkeypatch):

    books = {
        "YES_A": _book(0.28, 0.32, 400.0, 200.0),
        "YES_B": _book(0.27, 0.31, 500.0, 200.0),
        "YES_C": _book(0.29, 0.33, 450.0, 200.0),
    }
    detector = ComboArbDetector(books)
    cluster = ArbCluster(
        event_id="EVENT-1",
        legs=[
            _market("MKT_A", "YES_A", "NO_A", "Outcome A", liquidity_usd=800.0),
            _market("MKT_B", "YES_B", "NO_B", "Outcome B", liquidity_usd=600.0),
            _market("MKT_C", "YES_C", "NO_C", "Outcome C", liquidity_usd=900.0),
        ],
    )

    log_calls: list[tuple[str, dict[str, object]]] = []
    timestamps = iter([1000.0, 1010.0, 1020.0])

    monkeypatch.setattr(combinatorial_arb.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(
        combinatorial_arb.log,
        "info",
        lambda event, **kwargs: log_calls.append((event, kwargs)),
    )

    assert detector.evaluate_cluster(cluster, wallet_balance=1000.0) is None
    assert detector.evaluate_cluster(cluster, wallet_balance=1000.0) is None
    assert [event for event, _ in log_calls] == ["combo_arb_guardrail_rejected"]


def test_combo_arb_guardrail_logs_again_when_sum_bids_moves_by_more_than_five_points(monkeypatch):

    books = {
        "YES_A": _book(0.28, 0.32, 400.0, 200.0),
        "YES_B": _book(0.27, 0.31, 500.0, 200.0),
        "YES_C": _book(0.29, 0.33, 450.0, 200.0),
    }
    detector = ComboArbDetector(books)
    cluster = ArbCluster(
        event_id="EVENT-1",
        legs=[
            _market("MKT_A", "YES_A", "NO_A", "Outcome A", liquidity_usd=800.0),
            _market("MKT_B", "YES_B", "NO_B", "Outcome B", liquidity_usd=600.0),
            _market("MKT_C", "YES_C", "NO_C", "Outcome C", liquidity_usd=900.0),
        ],
    )

    log_calls: list[tuple[str, dict[str, object]]] = []
    timestamps = iter([1000.0, 1005.0, 1010.0, 1015.0])

    monkeypatch.setattr(combinatorial_arb.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(
        combinatorial_arb.log,
        "info",
        lambda event, **kwargs: log_calls.append((event, kwargs)),
    )

    assert detector.evaluate_cluster(cluster, wallet_balance=1000.0) is None
    books["YES_A"].on_book_snapshot(
        {
            "bids": [{"price": "0.22", "size": "400.0"}],
            "asks": [{"price": "0.32", "size": "200.0"}],
            "timestamp": 2,
        }
    )
    synced_update = books["YES_A"]._last_update
    books["YES_B"]._last_update = synced_update
    books["YES_C"]._last_update = synced_update

    assert detector.evaluate_cluster(cluster, wallet_balance=1000.0) is None
    assert [event for event, _ in log_calls] == [
        "combo_arb_guardrail_rejected",
        "combo_arb_guardrail_rejected",
    ]


def test_combo_arb_pauses_on_toxic_maker_leg_ofi():

    books = {
        "YES_A": _book(0.30, 0.34, 400.0, 200.0),
        "YES_B": _book(0.28, 0.38, 200.0, 3800.0),
        "YES_C": _book(0.29, 0.31, 450.0, 200.0),
    }
    detector = ComboArbDetector(books)
    cluster = ArbCluster(
        event_id="EVENT-1",
        legs=[
            _market("MKT_A", "YES_A", "NO_A", "Outcome A", liquidity_usd=800.0),
            _market("MKT_B", "YES_B", "NO_B", "Outcome B", liquidity_usd=600.0),
            _market("MKT_C", "YES_C", "NO_C", "Outcome C", liquidity_usd=900.0),
        ],
    )

    signal = detector.evaluate_cluster(cluster, wallet_balance=1000.0)

    assert signal is None
    assert detector._maker_leg_ofi_paused["YES_B"] is True


def test_combo_arb_suppresses_cluster_when_leg_books_are_desynced():
    sync_blocks: list[float] = []

    books = {
        "YES_A": _book(0.30, 0.34, 400.0, 200.0, timestamp=1.0),
        "YES_B": _book(0.28, 0.38, 500.0, 200.0, timestamp=1.45),
        "YES_C": _book(0.29, 0.31, 450.0, 200.0, timestamp=1.0),
    }
    detector = ComboArbDetector(books, on_sync_block=lambda assessment: sync_blocks.append(assessment.delta_ms))
    books["YES_A"]._last_update = 1.0
    books["YES_B"]._last_update = 1.45
    books["YES_C"]._last_update = 1.0
    cluster = ArbCluster(
        event_id="EVENT-1",
        legs=[
            _market("MKT_A", "YES_A", "NO_A", "Outcome A", liquidity_usd=800.0),
            _market("MKT_B", "YES_B", "NO_B", "Outcome B", liquidity_usd=600.0),
            _market("MKT_C", "YES_C", "NO_C", "Outcome C", liquidity_usd=900.0),
        ],
    )

    assert detector.evaluate_cluster(cluster, wallet_balance=1000.0) is None
    assert len(sync_blocks) == 1
    assert sync_blocks[0] > 400.0


def test_combo_arb_resumes_after_toxic_ofi_stabilizes(monkeypatch):

    books = {
        "YES_A": _book(0.30, 0.34, 400.0, 200.0),
        "YES_B": _book(0.28, 0.38, 200.0, 3800.0),
        "YES_C": _book(0.29, 0.31, 450.0, 200.0),
    }
    detector = ComboArbDetector(books)
    cluster = ArbCluster(
        event_id="EVENT-1",
        legs=[
            _market("MKT_A", "YES_A", "NO_A", "Outcome A", liquidity_usd=800.0),
            _market("MKT_B", "YES_B", "NO_B", "Outcome B", liquidity_usd=600.0),
            _market("MKT_C", "YES_C", "NO_C", "Outcome C", liquidity_usd=900.0),
        ],
    )

    log_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        combinatorial_arb.log,
        "info",
        lambda event, **kwargs: log_calls.append((event, kwargs)),
    )

    books["YES_A"]._last_update = 1.0
    books["YES_B"]._last_update = 1.0
    books["YES_C"]._last_update = 1.0

    assert detector.evaluate_cluster(cluster, wallet_balance=1000.0) is None

    books["YES_B"].on_book_snapshot(
        {
            "bids": [{"price": "0.28", "size": "3800.0"}],
            "asks": [{"price": "0.38", "size": "200.0"}],
            "timestamp": 2,
        }
    )
    books["YES_A"]._last_update = 5.5
    books["YES_B"]._last_update = 5.5
    books["YES_C"]._last_update = 5.5

    signal = detector.evaluate_cluster(cluster, wallet_balance=1000.0)

    assert signal is not None
    assert signal.maker_leg is not None
    assert signal.maker_leg.yes_token_id == "YES_B"
    assert detector._maker_leg_ofi_paused["YES_B"] is False
    resumed = detector.pop_recent_resume("EVENT-1")
    assert resumed is not None
    assert resumed.maker_leg == "YES_B"
    assert detector.pop_recent_resume("EVENT-1") is None
    assert [event for event, _ in log_calls] == [
        "combo_arb_ofi_paused",
        "combo_arb_ofi_resumed",
        "combo_arb_signal",
    ]


def test_combo_arb_widens_required_edge_for_latency_option_premium():

    books = {
        "YES_A": _book(0.31, 0.33, 400.0, 200.0),
        "YES_B": _book(0.31, 0.40, 500.0, 200.0),
        "YES_C": _book(0.32, 0.34, 450.0, 200.0),
    }
    aggregators = {
        "YES_A": FakeAgg(sigma=0.40, current_price=0.33),
        "YES_B": FakeAgg(sigma=0.02, current_price=0.40),
        "YES_C": FakeAgg(sigma=0.40, current_price=0.34),
    }
    detector = ComboArbDetector(books, aggregators=aggregators)
    cluster = ArbCluster(
        event_id="EVENT-1",
        legs=[
            _market("MKT_A", "YES_A", "NO_A", "Outcome A", liquidity_usd=800.0),
            _market("MKT_B", "YES_B", "NO_B", "Outcome B", liquidity_usd=600.0),
            _market("MKT_C", "YES_C", "NO_C", "Outcome C", liquidity_usd=900.0),
        ],
    )

    signal = detector.evaluate_cluster(cluster, wallet_balance=1000.0)

    assert signal is None