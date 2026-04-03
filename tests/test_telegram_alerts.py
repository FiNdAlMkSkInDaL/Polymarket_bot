from __future__ import annotations

from types import SimpleNamespace

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
            "sync_gate_counters": {
                "contagion_sync_blocks": 2,
                "si9_sync_blocks": 1,
                "si10_sync_blocks": 3,
            },
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
    assert "Sync gate: contagion=2" in sent[0]
    assert "MKT-A" in sent[0]
    assert "SELL tox=0.91" in sent[0]


@pytest.mark.asyncio
async def test_stats_update_formats_sync_gate_counters(monkeypatch):
    alerter = TelegramAlerter(bot_token="token", chat_id="chat")
    sent: list[str] = []

    async def _capture(message: str, parse_mode: str = "HTML") -> None:
        del parse_mode
        sent.append(message)

    monkeypatch.setattr(alerter, "send", _capture)

    await alerter.notify_stats(
        {
            "total_trades": 12,
            "sync_gate_counters": {
                "contagion_sync_blocks": 4,
                "si9_sync_blocks": 5,
                "si10_sync_blocks": 6,
            },
        }
    )

    assert len(sent) == 1
    assert "sync_gate: contagion=4 | si9=5 | si10=6" in sent[0]


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


@pytest.mark.asyncio
async def test_shield_paper_update_formats_summary(monkeypatch):
    alerter = TelegramAlerter(bot_token="token", chat_id="chat")
    sent: list[str] = []

    async def _capture(message: str, parse_mode: str = "HTML") -> None:
        del parse_mode
        sent.append(message)

    monkeypatch.setattr(alerter, "send", _capture)

    await alerter.notify_shield_paper_update(
        {
            "active_targets_loaded": 89,
            "submitted_orders": 7,
            "skipped_existing": 2,
            "rejected_orders": 1,
            "paper_intercepted_payloads": 7,
            "submitted_notional_usd": "350.00",
            "category_counts": {"sports": 27, "politics": 18, "business": 9},
            "submitted": [
                {"question": "Will BTC hit 120k?"},
                {"question": "Will Team A win the league?"},
            ],
        }
    )

    assert len(sent) == 1
    assert "<b>SHIELD PAPER</b>" in sent[0]
    assert "Paper bids staged: 7" in sent[0]
    assert "Planned notional: $350.00" in sent[0]
    assert "sports:27" in sent[0]
    assert "Will BTC hit 120k?" in sent[0]


@pytest.mark.asyncio
async def test_sword_paper_update_formats_scan_and_launch(monkeypatch):
    alerter = TelegramAlerter(bot_token="token", chat_id="chat")
    sent: list[str] = []

    async def _capture(message: str, parse_mode: str = "HTML") -> None:
        del parse_mode
        sent.append(message)

    monkeypatch.setattr(alerter, "send", _capture)

    await alerter.notify_sword_paper_update(
        scan_summary={
            "executable_strips": 2,
            "grouped_events_considered": 14,
            "targets": [
                {
                    "event_title": "Fed funds range by June?",
                    "recommended_action": "BUY_YES_STRIP",
                    "execution_edge_vs_fair_value": -0.031,
                    "strip_executable_notional_usd": 42.5,
                }
            ],
        },
        launch_summary={
            "targets_loaded": 2,
            "paper_intercepted_payloads": 6,
            "status_counts": {"SUBMITTED": 2},
        },
    )

    assert len(sent) == 1
    assert "<b>SWORD PAPER</b>" in sent[0]
    assert "Executable strips: 2" in sent[0]
    assert "Intercepted legs: 6" in sent[0]
    assert "BUY_YES_STRIP" in sent[0]


@pytest.mark.asyncio
async def test_pipeline_failure_alert(monkeypatch):
    alerter = TelegramAlerter(bot_token="token", chat_id="chat")
    sent: list[str] = []

    async def _capture(message: str, parse_mode: str = "HTML") -> None:
        del parse_mode
        sent.append(message)

    monkeypatch.setattr(alerter, "send", _capture)

    await alerter.notify_pipeline_failure("SHIELD", "refresh", "gamma timeout")

    assert len(sent) == 1
    assert "SHIELD Pipeline Failure" in sent[0]
    assert "gamma timeout" in sent[0]


@pytest.mark.asyncio
async def test_send_checked_returns_false_when_disabled():
    alerter = TelegramAlerter(bot_token="", chat_id="")
    assert await alerter.send_checked("hello") is False


@pytest.mark.asyncio
async def test_send_checked_returns_true_on_success(monkeypatch):
    alerter = TelegramAlerter(bot_token="token", chat_id="chat")

    class _Client:
        is_closed = False

        async def post(self, url: str, json: dict) -> SimpleNamespace:
            del url, json
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(alerter, "_get_client", lambda: _Client())
    assert await alerter.send_checked("hello") is True