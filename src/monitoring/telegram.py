"""Telegram alert bot — sends trade notifications and accepts a /kill command."""

from __future__ import annotations

import html as _html
import time
from collections import deque
from typing import Any

import httpx

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

# Telegram rate limit: max 30 messages per minute
_RATE_LIMIT_WINDOW_S = 60.0
_RATE_LIMIT_MAX = 30


class TelegramAlerter:
    """Lightweight async Telegram notification sender.

    Uses a shared ``httpx.AsyncClient`` for connection pooling and
    enforces a rate limit of 30 messages per minute to avoid Telegram
    429 errors.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id
        self._enabled = bool(self.bot_token and self.chat_id)
        self._client: httpx.AsyncClient | None = None
        self._send_times: deque[float] = deque(maxlen=_RATE_LIMIT_MAX)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and reuse a single httpx client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _send_impl(self, message: str, parse_mode: str = "HTML") -> bool:
        if not self._enabled:
            return False

        # Rate limiting: drop if we've sent too many messages recently
        now = time.monotonic()
        while self._send_times and (now - self._send_times[0]) > _RATE_LIMIT_WINDOW_S:
            self._send_times.popleft()
        if len(self._send_times) >= _RATE_LIMIT_MAX:
            log.warning("telegram_rate_limited", dropped_chars=len(message))
            return False

        # Truncate to Telegram's 4096 char limit
        if len(message) > 4096:
            message = message[:4090] + "\n..."

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }

        try:
            client = self._get_client()
            resp = await client.post(url, json=payload)
            self._send_times.append(now)
            if resp.status_code != 200:
                log.warning("telegram_send_failed", status=resp.status_code)
                return False
            return True
        except Exception as exc:
            log.warning("telegram_send_error", error=str(exc))
            return False

    async def send(self, message: str, parse_mode: str = "HTML") -> None:
        """Send a text message to the configured Telegram chat."""
        await self._send_impl(message, parse_mode=parse_mode)

    async def send_checked(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message and return whether Telegram acknowledged it."""
        return await self._send_impl(message, parse_mode=parse_mode)

    # ── Convenience methods ─────────────────────────────────────────────────
    async def notify_signal(self, market: str, zscore: float, v_ratio: float) -> None:
        safe = _html.escape(market[:60])
        await self.send(
            f"🔔 <b>Panic Signal</b>\n"
            f"Market: <code>{safe}</code>\n"
            f"Z-score: {zscore:.2f}  |  Vol ratio: {v_ratio:.1f}×"
        )

    async def notify_entry(
        self, pos_id: str, market: str, price: float, size: float, target: float
    ) -> None:
        safe = _html.escape(market[:40])
        await self.send(
            f"📥 <b>Entry Filled</b>\n"
            f"Pos: {pos_id}  |  {safe}\n"
            f"Buy NO @ {price:.2f}¢  ×{size:.1f}\n"
            f"Target: {target:.2f}¢"
        )

    async def notify_exit(
        self,
        pos_id: str,
        entry: float,
        exit_p: float,
        pnl: float,
        reason: str,
        smart_passive_counters: dict[str, int] | None = None,
    ) -> None:
        emoji = "✅" if pnl > 0 else "❌"
        counters_block = self._format_smart_passive_counters(
            smart_passive_counters,
            include_zero=False,
        )
        await self.send(
            f"{emoji} <b>Position Closed</b>\n"
            f"Pos: {pos_id}\n"
            f"Entry: {entry:.2f}¢ → Exit: {exit_p:.2f}¢\n"
            f"PnL: {pnl:+.2f}¢  ({reason})"
            f"{counters_block}"
        )

    async def notify_stats(self, stats: dict) -> None:
        lines = [f"📊 <b>Stats Update</b>"]
        for k, v in stats.items():
            if k == "smart_passive_counters" and isinstance(v, dict):
                lines.append(
                    "  smart_passive: "
                    f"started={int(v.get('smart_passive_started', 0))} | "
                    f"maker_filled={int(v.get('maker_filled', 0))} | "
                    f"fallback={int(v.get('fallback_triggered', 0))}"
                )
                continue
            if k == "sync_gate_counters" and isinstance(v, dict):
                lines.append(
                    "  sync_gate: "
                    f"contagion={int(v.get('contagion_sync_blocks', 0))} | "
                    f"si9={int(v.get('si9_sync_blocks', 0))} | "
                    f"si10={int(v.get('si10_sync_blocks', 0))}"
                )
                continue
            lines.append(f"  {k}: {v}")
        await self.send("\n".join(lines))

    async def notify_combo_arb_deferred(
        self,
        event_id: str,
        maker_leg: str,
        rolling_vi: float,
        threshold: float,
        *,
        current_vi: float = 0.0,
        question: str = "",
    ) -> None:
        safe_event = _html.escape(event_id[:24])
        safe_leg = _html.escape(maker_leg[:24])
        safe_question = _html.escape(question[:80])
        question_line = f"Market: {safe_question}\n" if safe_question else ""
        await self.send(
            f"⏸️ <b>SI-9 Deferred</b>\n"
            f"Event: <code>{safe_event}</code>\n"
            f"{question_line}"
            f"Maker leg: <code>{safe_leg}</code>\n"
            f"Reason: toxic OFI wave\n"
            f"Rolling VI: {rolling_vi:.3f}  |  Current VI: {current_vi:.3f}\n"
            f"Threshold: {threshold:.3f}"
        )

    async def notify_combo_arb_resumed(
        self,
        event_id: str,
        maker_leg: str,
        defer_count: int,
        *,
        question: str = "",
    ) -> None:
        safe_event = _html.escape(event_id[:24])
        safe_leg = _html.escape(maker_leg[:24])
        safe_question = _html.escape(question[:80])
        question_line = f"Market: {safe_question}\n" if safe_question else ""
        await self.send(
            f"▶️ <b>SI-9 Resumed</b>\n"
            f"Event: <code>{safe_event}</code>\n"
            f"{question_line}"
            f"Maker leg: <code>{safe_leg}</code>\n"
            f"Deferred scans cleared: {defer_count}"
        )

    async def notify_bayesian_arb_signal(
        self,
        relationship_id: str,
        *,
        label: str,
        bound_title: str,
        bound_expression: str,
        observed_yes_prices: dict[str, float],
        traded_leg_prices: dict[str, dict[str, float | str]],
        shares: float,
        edge_cents: float,
        gross_edge_cents: float,
        spread_cost_cents: float,
        taker_fee_cents: float,
        net_ev_usd: float,
        annualized_yield: float,
        days_to_resolution: float,
        collateral_usd: float,
    ) -> None:
        safe_relationship = _html.escape(relationship_id[:24])
        safe_label = _html.escape(label[:80])
        safe_bound_title = _html.escape(bound_title)
        safe_expression = _html.escape(bound_expression)
        observed_block = (
            f"A YES: {float(observed_yes_prices.get('base_a_yes', 0.0)):.4f}\n"
            f"B YES: {float(observed_yes_prices.get('base_b_yes', 0.0)):.4f}\n"
            f"Joint YES: {float(observed_yes_prices.get('joint_yes', 0.0)):.4f}"
        )
        leg_lines: list[str] = []
        for asset_id, leg in traded_leg_prices.items():
            leg_lines.append(
                f"<code>{_html.escape(asset_id[:20])}</code> "
                f"{_html.escape(str(leg.get('role', '')))} "
                f"{_html.escape(str(leg.get('trade_side', '')))} "
                f"bid={float(leg.get('best_bid', 0.0)):.4f} "
                f"ask={float(leg.get('best_ask', 0.0)):.4f} "
                f"target={float(leg.get('target_price', 0.0)):.4f} "
                f"fee={int(float(leg.get('fee_bps', 0.0) or 0.0))}bps"
            )

        await self.send(
            f"📐 <b>SI-10 Bayesian Arb</b>\n"
            f"Relationship: <code>{safe_relationship}</code>\n"
            f"Label: {safe_label}\n"
            f"{safe_bound_title}\n"
            f"Math: {safe_expression}\n"
            f"Observed YES prices:\n{observed_block}\n"
            f"Traded legs:\n" + "\n".join(leg_lines) + "\n"
            f"Shares: {shares:.2f}  |  Net edge: {edge_cents:.2f}¢  |  Collateral: ${collateral_usd:.2f}\n"
            f"Gross: {gross_edge_cents:.2f}¢  |  Spread: {spread_cost_cents:.2f}¢  |  Fees: {taker_fee_cents:.2f}¢\n"
            f"Net EV: ${net_ev_usd:.4f}  |  Annualized: {annualized_yield * 100.0:.2f}%  |  Horizon: {days_to_resolution:.1f}d"
        )

    async def notify_toxicity_rankings(self, rankings: list[dict[str, Any]]) -> None:
        if not rankings:
            return

        lines = ["☣️ <b>Top Market Toxicity</b>"]
        for idx, row in enumerate(rankings[:5], 1):
            question = _html.escape(str(row.get("question", ""))[:48])
            condition_id = _html.escape(str(row.get("condition_id", ""))[:16])
            lines.append(
                f"{idx}. <code>{condition_id}</code> {question}\n"
                f"   {row.get('dominant_side', 'BUY')} tox={float(row.get('toxicity_index', 0.0)):.2f}  "
                f"evap={float(row.get('depth_evaporation', 0.0)):.2f}  "
                f"sweep={float(row.get('sweep_ratio', 0.0)):.2f}"
            )
        await self.send("\n".join(lines))

    async def notify_rpe_signal(
        self,
        market_id: str,
        model_prob: float,
        market_price: float,
        direction: str,
        confidence: float,
        shadow: bool,
        question: str = "",
        *,
        calibration_footer: str = "",
    ) -> None:
        mode = "👻 SHADOW" if shadow else "🎯 LIVE"
        arrow = "⬇️" if direction == "buy_no" else "⬆️"
        safe_id = _html.escape(market_id)
        question_line = f"<b>{_html.escape(question[:120])}</b>\n" if question else ""
        footer = _html.escape(calibration_footer) if calibration_footer else ""
        await self.send(
            f"{mode} <b>RPE Signal</b> {arrow}\n"
            f"{question_line}"
            f"ID: <code>{safe_id}</code>\n"
            f"Model: {model_prob:.3f}  |  Market: {market_price:.3f}\n"
            f"Direction: {direction}  |  Confidence: {confidence:.2f}"
            f"{footer}"
        )

    async def notify_rpe_calibration_report(self, tracker: Any) -> None:
        """Send a calibration dashboard for the RPE.

        Parameters
        ----------
        tracker:
            An ``RPECalibrationTracker`` instance.  Accepts ``Any`` to
            avoid a circular import.
        """
        stats = tracker.calibration_summary()
        total = stats.get("total_signals", 0)
        live = stats.get("live_signals", 0)
        shadow = stats.get("shadow_signals", 0)
        resolved = stats.get("resolved", 0)
        brier = stats.get("brier_score", "n/a")
        logloss = stats.get("log_loss", "n/a")
        accuracy = stats.get("direction_accuracy", "n/a")

        if isinstance(accuracy, float):
            accuracy = f"{accuracy:.1%}"

        await self.send(
            f"📊 <b>RPE Calibration Report</b>\n"
            f"Total signals: {total} (live: {live}, shadow: {shadow})\n"
            f"Resolved: {resolved}\n"
            f"Brier score: {brier}\n"
            f"Log-loss: {logloss}\n"
            f"Direction accuracy: {accuracy}"
        )

    async def notify_pce_dashboard(self, data: dict) -> None:
        """Send PCE correlation dashboard summary."""
        var_val = data.get("portfolio_var", 0.0)
        threshold = data.get("threshold", 15.0)
        pairs = data.get("top_correlated_pairs", [])
        n_pos = data.get("open_positions", 0)
        gross = data.get("gross_exposure", 0.0)
        net = data.get("net_exposure", 0.0)
        shadow = data.get("shadow_mode", True)
        mode = "👻 SHADOW" if shadow else "🎯 LIVE"

        pair_lines = ""
        for i, p in enumerate(pairs[:3], 1):
            a = _html.escape(p.get("market_a", "")[:40])
            b = _html.escape(p.get("market_b", "")[:40])
            c = p.get("correlation", 0.0)
            pair_lines += f"  {i}. <code>{a}</code>\n     ↔ <code>{b}</code>: {c:.3f}\n"

        if not pair_lines:
            pair_lines = "  (no pairs tracked)\n"

        await self.send(
            f"🔗 {mode} <b>PCE Dashboard</b>\n"
            f"Portfolio VaR: ${var_val:.2f} / ${threshold:.2f}\n"
            f"Top correlated:\n{pair_lines}"
            f"Positions: {n_pos}  |  Gross: ${gross:.2f}  |  Net: ${net:.2f}"
        )

    async def notify_contagion_matrix(self, data: dict[str, Any]) -> None:
        """Send a structured Domino shadow/live matrix alert."""
        mode = "👻 SHADOW" if data.get("shadow", True) else "🎯 LIVE"
        if data.get("suppressed"):
            mode = f"{mode} SUPPRESSED"

        leader_id = _html.escape(str(data.get("leader_market_id", ""))[:24])
        lagger_id = _html.escape(str(data.get("lagging_market_id", ""))[:24])
        leader_q = _html.escape(str(data.get("leader_question", ""))[:72])
        lagger_q = _html.escape(str(data.get("lagging_question", ""))[:72])
        question_block = ""
        if leader_q or lagger_q:
            question_block = (
                f"Leader: {leader_q or leader_id}\n"
                f"Lagger: {lagger_q or lagger_id}\n"
            )

        suppression = ""
        if data.get("suppressed"):
            suppression = f"\nSuppressed: {_html.escape(str(data.get('suppression_reason', 'unknown')))}"

        await self.send(
            f"🧩 {mode} <b>Domino Matrix</b>\n"
            f"{question_block}"
            f"Leader <code>{leader_id}</code> {data.get('leader_direction', '')} tox-excess="
            f"{float(data.get('leader_toxicity_excess', 0.0)):.3f}\n"
            f"Lagger <code>{lagger_id}</code> corr={float(data.get('correlation', 0.0)):.3f}  |  "
            f"theme={_html.escape(str(data.get('thematic_group', ''))[:24])}\n"
            f"Shift math: Δleader={float(data.get('leader_price_shift_cents', 0.0)):+.2f}¢  ×  "
            f"ρ={float(data.get('correlation', 0.0)):.3f}  →  Δfair={float(data.get('expected_shift_cents', 0.0)):+.2f}¢\n"
            f"Fair={float(data.get('fair_value', 0.0)):.3f}  |  Market={float(data.get('market_price', 0.0)):.3f}  |  "
            f"Edge={float(data.get('edge_cents', 0.0)):+.2f}¢\n"
            f"Cross slip={float(data.get('cross_spread_slip_cents', 0.0)):.2f}¢  |  "
            f"Last trade age={float(data.get('last_trade_age_s', -1.0)):.1f}s"
            f"{suppression}"
        )

    async def notify_paper_summary(
        self,
        stats: dict,
        uptime_h: float,
        *,
        toxicity_rankings: list[dict[str, Any]] | None = None,
    ) -> None:
        """Send a formatted paper trade performance summary.

        Parameters
        ----------
        stats:
            Dict from ``TradeStore.get_stats()`` with keys like
            ``total_trades``, ``win_rate``, ``avg_pnl``, ``total_pnl``,
            ``best_trade``, ``worst_trade``.
        uptime_h:
            Bot uptime in hours — provides context for trade frequency.
        """
        total = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0.0)
        avg_pnl = stats.get("avg_pnl", 0.0)
        total_pnl = stats.get("total_pnl", 0.0)
        best = stats.get("best_trade", 0.0)
        worst = stats.get("worst_trade", 0.0)
        counters_block = self._format_smart_passive_counters(
            stats.get("smart_passive_counters"),
            include_zero=True,
        )
        sync_gate_block = self._format_sync_gate_counters(
            stats.get("sync_gate_counters"),
            include_zero=True,
        )

        emoji = "📈" if total_pnl >= 0 else "📉"
        tph = total / max(0.01, uptime_h)
        toxicity_lines = ""
        if toxicity_rankings:
            ranked_lines = []
            for idx, row in enumerate(toxicity_rankings[:5], 1):
                condition_id = _html.escape(str(row.get("condition_id", ""))[:16])
                question = _html.escape(str(row.get("question", ""))[:36])
                ranked_lines.append(
                    f"{idx}. <code>{condition_id}</code> {question}\n"
                    f"   {row.get('dominant_side', 'BUY')} tox={float(row.get('toxicity_index', 0.0)):.2f}  "
                    f"evap={float(row.get('depth_evaporation', 0.0)):.2f}  "
                    f"sweep={float(row.get('sweep_ratio', 0.0)):.2f}"
                )
            toxicity_lines = "\n☣️ <b>Top Toxicity</b>\n" + "\n".join(ranked_lines)

        await self.send(
            f"{emoji} <b>Paper Trade Summary</b>\n"
            f"Trades: {total}  ({tph:.1f}/hr)\n"
            f"Win rate: {win_rate:.1%}\n"
            f"Avg PnL: {avg_pnl:+.2f}¢  |  Total: {total_pnl:+.2f}¢\n"
            f"Best: {best:+.2f}¢  |  Worst: {worst:+.2f}¢\n"
            f"Uptime: {uptime_h:.1f}h"
            f"{counters_block}"
            f"{sync_gate_block}"
            f"{toxicity_lines}"
        )

    async def notify_shield_paper_update(self, summary: dict[str, Any]) -> None:
        active_targets = int(summary.get("active_targets_loaded", 0) or 0)
        submitted_orders = int(summary.get("submitted_orders", 0) or 0)
        skipped_existing = int(summary.get("skipped_existing", 0) or 0)
        rejected_orders = int(summary.get("rejected_orders", 0) or 0)
        intercepted = int(summary.get("paper_intercepted_payloads", 0) or 0)
        submitted_notional = float(summary.get("submitted_notional_usd", 0.0) or 0.0)
        category_counts = summary.get("category_counts") if isinstance(summary.get("category_counts"), dict) else {}
        top_categories = sorted(category_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:4]
        category_line = ", ".join(f"{_html.escape(str(name))}:{int(count)}" for name, count in top_categories) if top_categories else "n/a"

        submitted_rows = summary.get("submitted") if isinstance(summary.get("submitted"), list) else []
        sample_questions = []
        for row in submitted_rows[:3]:
            if not isinstance(row, dict):
                continue
            question = _html.escape(str(row.get("question", ""))[:72])
            if question:
                sample_questions.append(f"- {question}")
        questions_block = "\n" + "\n".join(sample_questions) if sample_questions else ""

        await self.send(
            f"🛡️ <b>SHIELD PAPER</b>\n"
            f"Active targets: {active_targets}\n"
            f"Paper bids staged: {submitted_orders}  |  Intercepted: {intercepted}\n"
            f"Skipped existing: {skipped_existing}  |  Rejected: {rejected_orders}\n"
            f"Planned notional: ${submitted_notional:.2f}\n"
            f"Top categories: {category_line}"
            f"{questions_block}"
        )

    async def notify_sword_paper_update(
        self,
        *,
        scan_summary: dict[str, Any],
        launch_summary: dict[str, Any] | None = None,
    ) -> None:
        executable_strips = int(scan_summary.get("executable_strips", 0) or 0)
        grouped_events = int(scan_summary.get("grouped_events_considered", 0) or 0)
        targets = scan_summary.get("targets") if isinstance(scan_summary.get("targets"), list) else []

        top_lines: list[str] = []
        for idx, row in enumerate(targets[:3], 1):
            if not isinstance(row, dict):
                continue
            title = _html.escape(str(row.get("event_title", ""))[:60])
            action = _html.escape(str(row.get("recommended_action", ""))[:24])
            edge = float(row.get("execution_edge_vs_fair_value", 0.0) or 0.0)
            notional = float(row.get("strip_executable_notional_usd", 0.0) or 0.0)
            top_lines.append(f"{idx}. {title}  |  {action}  |  edge={edge:+.4f}  |  depth=${notional:.2f}")
        top_block = "\n" + "\n".join(top_lines) if top_lines else ""

        launch_block = ""
        if launch_summary is not None:
            status_counts = launch_summary.get("status_counts") if isinstance(launch_summary.get("status_counts"), dict) else {}
            intercepted = int(launch_summary.get("paper_intercepted_payloads", 0) or 0)
            targets_loaded = int(launch_summary.get("targets_loaded", 0) or 0)
            formatted_counts = ", ".join(f"{_html.escape(str(name))}:{int(count)}" for name, count in sorted(status_counts.items())) or "n/a"
            launch_block = (
                "\n"
                f"Launch targets: {targets_loaded}  |  Intercepted legs: {intercepted}\n"
                f"Statuses: {formatted_counts}"
            )

        await self.send(
            f"⚔️ <b>SWORD PAPER</b>\n"
            f"Executable strips: {executable_strips}\n"
            f"Grouped events scanned: {grouped_events}"
            f"{launch_block}"
            f"{top_block}"
        )

    async def notify_pipeline_failure(self, strategy: str, stage: str, message: str) -> None:
        safe_strategy = _html.escape(strategy[:24])
        safe_stage = _html.escape(stage[:32])
        safe_message = _html.escape(message[:1200])
        await self.send(
            f"🚨 <b>{safe_strategy} Pipeline Failure</b>\n"
            f"Stage: <code>{safe_stage}</code>\n"
            f"{safe_message}"
        )

    @staticmethod
    def _format_smart_passive_counters(
        counters: dict[str, int] | None,
        *,
        include_zero: bool,
    ) -> str:
        if not counters:
            return ""

        started = int(counters.get("smart_passive_started", 0) or 0)
        maker_filled = int(counters.get("maker_filled", 0) or 0)
        fallback = int(counters.get("fallback_triggered", 0) or 0)
        if not include_zero and started == 0 and maker_filled == 0 and fallback == 0:
            return ""

        return (
            "\n"
            f"Smart-passive: {started} started  |  "
            f"{maker_filled} maker-filled  |  "
            f"{fallback} fallback"
        )

    @staticmethod
    def _format_sync_gate_counters(
        counters: dict[str, int] | None,
        *,
        include_zero: bool,
    ) -> str:
        if not counters:
            return ""

        contagion = int(counters.get("contagion_sync_blocks", 0) or 0)
        si9 = int(counters.get("si9_sync_blocks", 0) or 0)
        si10 = int(counters.get("si10_sync_blocks", 0) or 0)
        if not include_zero and contagion == 0 and si9 == 0 and si10 == 0:
            return ""

        return (
            "\n"
            f"Sync gate: contagion={contagion}  |  "
            f"si9={si9}  |  "
            f"si10={si10}"
        )

    async def notify_kill(self) -> None:
        await self.send("🛑 <b>KILL SWITCH ACTIVATED</b> — all orders cancelled, bot stopping.")

    # ── Shadow Performance Tracker: graduation alert ────────────────────
    async def shadow_graduation_alert(
        self,
        signal_source: str,
        stats: dict,
    ) -> None:
        """Send an HTML-formatted alert when a shadow strategy passes go-live criteria."""
        total = stats.get("total_trades", 0)
        wr = stats.get("win_rate", 0.0)
        ev = stats.get("expectancy_cents", 0.0)
        total_pnl = stats.get("total_pnl_cents", 0.0)
        max_dd = stats.get("max_drawdown_cents", 0.0)

        safe_source = _html.escape(signal_source)
        await self.send(
            f"🎓 <b>SHADOW STRATEGY GRADUATION</b>\n\n"
            f"Signal: <code>{safe_source}</code>\n"
            f"Status: <b>READY FOR DEPLOYMENT</b>\n\n"
            f"📊 <b>Counterfactual Performance</b>\n"
            f"  Trades: {total}\n"
            f"  Win rate: {wr:.1%}\n"
            f"  Expectancy: {ev:+.2f}¢/trade\n"
            f"  Total PnL: {total_pnl:+.2f}¢\n"
            f"  Max DD: {max_dd:.2f}¢\n\n"
            f"✅ This shadow strategy has proved its statistical edge "
            f"and is ready for live capital allocation."
        )

    async def check_shadow_graduations(self, trade_store: object) -> None:
        """Daily cron-style check: evaluate all shadow strategies and
        alert on any that cross the go-live threshold.

        Parameters
        ----------
        trade_store:
            A ``TradeStore`` instance with ``get_all_shadow_sources()``
            and ``passes_shadow_go_live()`` methods.
        """
        try:
            sources = await trade_store.get_all_shadow_sources()  # type: ignore[union-attr]
            for source in sources:
                ready, stats = await trade_store.passes_shadow_go_live(source)  # type: ignore[union-attr]
                if ready:
                    await self.shadow_graduation_alert(source, stats)
        except Exception as exc:
            log.warning("shadow_graduation_check_failed", error=str(exc))
