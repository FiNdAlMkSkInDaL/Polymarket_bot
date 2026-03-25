from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.data.market_discovery import MarketInfo
from src.data.orderbook import OrderbookTracker
from src.signals.bayesian_arb import BayesianArbCluster, BayesianArbDetector


def _market(
    condition_id: str,
    question: str,
    yes_token_id: str,
    no_token_id: str,
    *,
    end_date: datetime | None = None,
    tags: str = "",
) -> MarketInfo:
    return MarketInfo(
        condition_id=condition_id,
        question=question,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        daily_volume_usd=1000.0,
        end_date=end_date or datetime.now(timezone.utc) + timedelta(days=7),
        active=True,
        event_id=f"EV-{condition_id}",
        liquidity_usd=1000.0,
        tags=tags,
    )


def _book(best_bid: float, best_ask: float, bid_size: float = 200.0, ask_size: float = 200.0) -> OrderbookTracker:
    book = OrderbookTracker("unused")
    book.on_book_snapshot(
        {
            "bids": [{"price": str(best_bid), "size": str(bid_size)}],
            "asks": [{"price": str(best_ask), "size": str(ask_size)}],
            "timestamp": 1,
        }
    )
    return book


def test_bayesian_arb_detects_upper_bound_violation_with_yes_and_no_legs():
    base_a = _market("MKT_A", "Will A happen?", "YES_A", "NO_A")
    base_b = _market("MKT_B", "Will B happen?", "YES_B", "NO_B")
    joint = _market("MKT_AB", "Will A and B happen?", "YES_AB", "NO_AB")
    cluster = BayesianArbCluster(
        relationship_id="REL-AB",
        base_a=base_a,
        base_b=base_b,
        joint=joint,
        label="A/B joint",
    )
    books = {
        "YES_A": _book(0.40, 0.41),
        "NO_A": _book(0.59, 0.60),
        "YES_B": _book(0.60, 0.61),
        "NO_B": _book(0.39, 0.40),
        "YES_AB": _book(0.46, 0.47),
        "NO_AB": _book(0.53, 0.58),
    }

    signal = BayesianArbDetector(books).evaluate_cluster(cluster, wallet_balance=1000.0)

    assert signal is not None
    assert signal.violation_type == "upper_a"
    assert signal.maker_leg is not None
    assert {leg.trade_side for leg in signal.legs} == {"YES", "NO"}
    assert {leg.asset_id for leg in signal.legs} == {"YES_A", "NO_AB"}
    assert signal.hedge_weights == {leg.asset_id: 1.0 for leg in signal.legs}
    assert signal.raw_gap_cents == 6.0
    assert signal.total_collateral > 0.0
    assert signal.net_ev_usd > 0.0
    assert signal.annualized_yield > 0.15
    assert signal.days_to_resolution > 0.0
    assert signal.bound_title == "Upper Bound Violation"
    assert signal.bound_expression == "P(A ∩ B) = 0.4600 > min(P(A), P(B)) = 0.4000"
    assert signal.observed_yes_prices == {
        "base_a_yes": 0.4,
        "base_b_yes": 0.6,
        "joint_yes": 0.46,
    }


def test_bayesian_arb_detects_lower_bound_violation_with_three_leg_hedge():
    base_a = _market("MKT_A", "Will A happen?", "YES_A", "NO_A")
    base_b = _market("MKT_B", "Will B happen?", "YES_B", "NO_B")
    joint = _market("MKT_AB", "Will A and B happen?", "YES_AB", "NO_AB")
    cluster = BayesianArbCluster(
        relationship_id="REL-LOWER",
        base_a=base_a,
        base_b=base_b,
        joint=joint,
        label="A/B lower",
    )
    books = {
        "YES_A": _book(0.70, 0.71),
        "NO_A": _book(0.31, 0.35),
        "YES_B": _book(0.65, 0.66),
        "NO_B": _book(0.36, 0.37),
        "YES_AB": _book(0.20, 0.21, bid_size=300.0),
        "NO_AB": _book(0.79, 0.80),
    }

    signal = BayesianArbDetector(books).evaluate_cluster(cluster, wallet_balance=1000.0)

    assert signal is not None
    assert signal.violation_type == "lower"
    assert signal.maker_leg is not None
    assert len(signal.legs) == 3
    assert [leg.trade_side for leg in signal.legs].count("NO") == 2
    assert [leg.trade_side for leg in signal.legs].count("YES") == 1
    assert {leg.asset_id for leg in signal.legs} == {"YES_AB", "NO_A", "NO_B"}
    assert signal.raw_gap_cents == 15.0
    assert signal.edge_cents > 0.0
    assert signal.bound_title == "Lower Bound Violation"
    assert signal.bound_expression == "P(A ∩ B) = 0.2000 < P(A) + P(B) - 1 = 0.3500"


def test_bayesian_arb_suppresses_negative_net_ev_after_fees_and_spread():
    base_a = _market("MKT_A", "Will A happen?", "YES_A", "NO_A", tags="crypto")
    base_b = _market("MKT_B", "Will B happen?", "YES_B", "NO_B", tags="crypto")
    joint = _market("MKT_AB", "Will A and B happen?", "YES_AB", "NO_AB", tags="crypto")
    cluster = BayesianArbCluster(
        relationship_id="REL-NET",
        base_a=base_a,
        base_b=base_b,
        joint=joint,
        label="A/B net gate",
    )
    books = {
        "YES_A": _book(0.49, 0.50),
        "NO_A": _book(0.50, 0.51),
        "YES_B": _book(0.70, 0.71),
        "NO_B": _book(0.29, 0.30),
        "YES_AB": _book(0.52, 0.53),
        "NO_AB": _book(0.48, 0.56),
    }

    detector = BayesianArbDetector(
        books,
        fee_enabled_resolver=lambda market: True,
        fee_rate_bps_lookup=lambda asset_id: 1000,
    )

    assert detector.evaluate_cluster(cluster, wallet_balance=1000.0) is None


def test_bayesian_arb_rejects_low_annualized_yield():
    far_end = datetime.now(timezone.utc) + timedelta(days=365)
    base_a = _market("MKT_A", "Will A happen?", "YES_A", "NO_A", end_date=far_end)
    base_b = _market("MKT_B", "Will B happen?", "YES_B", "NO_B", end_date=far_end)
    joint = _market("MKT_AB", "Will A and B happen?", "YES_AB", "NO_AB", end_date=far_end)
    cluster = BayesianArbCluster(
        relationship_id="REL-YIELD",
        base_a=base_a,
        base_b=base_b,
        joint=joint,
        label="A/B annualized gate",
    )
    books = {
        "YES_A": _book(0.49, 0.50),
        "NO_A": _book(0.50, 0.51),
        "YES_B": _book(0.60, 0.61),
        "NO_B": _book(0.39, 0.40),
        "YES_AB": _book(0.50, 0.51),
        "NO_AB": _book(0.48, 0.49),
    }

    detector = BayesianArbDetector(books)

    assert detector.evaluate_cluster(cluster, wallet_balance=1000.0) is None