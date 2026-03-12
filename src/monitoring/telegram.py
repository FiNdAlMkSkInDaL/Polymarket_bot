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

    async def send(self, message: str, parse_mode: str = "HTML") -> None:
        """Send a text message to the configured Telegram chat."""
        if not self._enabled:
            return

        # Rate limiting: drop if we've sent too many messages recently
        now = time.monotonic()
        while self._send_times and (now - self._send_times[0]) > _RATE_LIMIT_WINDOW_S:
            self._send_times.popleft()
        if len(self._send_times) >= _RATE_LIMIT_MAX:
            log.warning("telegram_rate_limited", dropped_chars=len(message))
            return

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
        except Exception as exc:
            log.warning("telegram_send_error", error=str(exc))

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
        self, pos_id: str, entry: float, exit_p: float, pnl: float, reason: str
    ) -> None:
        emoji = "✅" if pnl > 0 else "❌"
        await self.send(
            f"{emoji} <b>Position Closed</b>\n"
            f"Pos: {pos_id}\n"
            f"Entry: {entry:.2f}¢ → Exit: {exit_p:.2f}¢\n"
            f"PnL: {pnl:+.2f}¢  ({reason})"
        )

    async def notify_stats(self, stats: dict) -> None:
        lines = [f"📊 <b>Stats Update</b>"]
        for k, v in stats.items():
            lines.append(f"  {k}: {v}")
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

    async def notify_paper_summary(self, stats: dict, uptime_h: float) -> None:
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

        emoji = "📈" if total_pnl >= 0 else "📉"
        tph = total / max(0.01, uptime_h)

        await self.send(
            f"{emoji} <b>Paper Trade Summary</b>\n"
            f"Trades: {total}  ({tph:.1f}/hr)\n"
            f"Win rate: {win_rate:.1%}\n"
            f"Avg PnL: {avg_pnl:+.2f}¢  |  Total: {total_pnl:+.2f}¢\n"
            f"Best: {best:+.2f}¢  |  Worst: {worst:+.2f}¢\n"
            f"Uptime: {uptime_h:.1f}h"
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
