from __future__ import annotations

import pytest

from src.monitoring.telegram import TelegramAlerter


@pytest.mark.asyncio
async def test_paper_summary_includes_top_toxicity_rankings(monkeypatch):
    alerter = TelegramAlerter(bot_token="token", chat_id="chat")
    sent: list[str] = []

    async def _capture(message: str, parse_mode: str = "HTML") -> None:
        del parse_mode
        sent.append(message)

    monkeypatch.setattr(alerter, "send", _capture)

    await alerter.notify_paper_summary(
        {
            "total_trades": 10,
            "win_rate": 0.6,
            "avg_pnl": 1.5,
            "total_pnl": 15.0,
            "best_trade": 5.0,
            "worst_trade": -2.0,
        },
        2.0,
        toxicity_rankings=[
            {
                "condition_id": "MKT-A",
                "question": "Will BTC close above 100k?",
                "dominant_side": "SELL",
                "toxicity_index": 0.91,
                "depth_evaporation": 0.40,
                "sweep_ratio": 0.55,
            },
            {
                "condition_id": "MKT-B",
                "question": "Will ETH outperform BTC?",
                "dominant_side": "BUY",
                "toxicity_index": 0.82,
                "depth_evaporation": 0.24,
                "sweep_ratio": 0.41,
            },
        ],
    )

    assert len(sent) == 1
    assert "<b>Paper Trade Summary</b>" in sent[0]
    assert "<b>Top Toxicity</b>" in sent[0]
    assert "MKT-A" in sent[0]
    assert "SELL tox=0.91" in sent[0]


@pytest.mark.asyncio
async def test_contagion_matrix_formats_domino_payload(monkeypatch):
    alerter = TelegramAlerter(bot_token="token", chat_id="chat")
    sent: list[str] = []

    async def _capture(message: str, parse_mode: str = "HTML") -> None:
        del parse_mode
        sent.append(message)

    monkeypatch.setattr(alerter, "send", _capture)

    await alerter.notify_contagion_matrix(
        {
            "shadow": True,
            "leader_market_id": "LEADER-1",
            "lagging_market_id": "LAGGER-1",
            "leader_question": "Will BTC break 100k?",
            "lagging_question": "Will ETH break 5k?",
            "leader_direction": "buy_yes",
            "leader_toxicity_excess": 0.125,
            "correlation": 0.74,
            "thematic_group": "crypto",
            "leader_price_shift_cents": 2.5,
            "expected_shift_cents": 1.8,
            "fair_value": 0.63,
            "market_price": 0.58,
            "edge_cents": 3.0,
            "cross_spread_slip_cents": 0.7,
            "last_trade_age_s": 24.0,
        }
    )

    assert len(sent) == 1
    assert "<b>Domino Matrix</b>" in sent[0]
    assert "LEADER-1" in sent[0]
    assert "LAGGER-1" in sent[0]
    assert "Δfair=+1.80¢" in sent[0]
    assert "Cross slip=0.70¢" in sent[0]


@pytest.mark.asyncio
async def test_bayesian_arb_alert_includes_bound_math(monkeypatch):
    alerter = TelegramAlerter(bot_token="token", chat_id="chat")
    sent: list[str] = []

    async def _capture(message: str, parse_mode: str = "HTML") -> None:
        del parse_mode
        sent.append(message)

    monkeypatch.setattr(alerter, "send", _capture)

    await alerter.notify_bayesian_arb_signal(
        "REL-1",
        label="Example relationship",
        bound_title="Upper Bound Violation",
        bound_expression="P(A ∩ B) = 0.4600 > min(P(A), P(B)) = 0.4000",
        observed_yes_prices={
            "base_a_yes": 0.4,
            "base_b_yes": 0.6,
            "joint_yes": 0.46,
        },
        traded_leg_prices={
            "YES_A": {
                "role": "base_a",
                "trade_side": "YES",
                "best_bid": 0.4,
                "best_ask": 0.41,
                "target_price": 0.4,
            },
            "NO_AB": {
                "role": "joint",
                "trade_side": "NO",
                "best_bid": 0.53,
                "best_ask": 0.58,
                "target_price": 0.58,
            },
        },
        shares=25.0,
        edge_cents=2.0,
        gross_edge_cents=6.0,
        spread_cost_cents=3.0,
        taker_fee_cents=1.0,
        net_ev_usd=0.5,
        annualized_yield=0.42,
        days_to_resolution=7.0,
        collateral_usd=24.5,
    )

    assert len(sent) == 1
    assert "Upper Bound Violation" in sent[0]
    assert "Math: P(A ∩ B) = 0.4600 &gt; min(P(A), P(B)) = 0.4000" in sent[0]
    assert "A YES: 0.4000" in sent[0]
    assert "Joint YES: 0.4600" in sent[0]
    assert "Gross: 6.00¢" in sent[0]
    assert "Net EV: $0.5000" in sent[0]
    assert "Annualized: 42.00%" in sent[0]