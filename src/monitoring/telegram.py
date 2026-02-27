"""
Telegram alert bot — sends trade notifications and accepts a /kill command.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


class TelegramAlerter:
    """Lightweight async Telegram notification sender.

    Uses httpx directly to avoid heavyweight telegram library at PoC stage.
    Supports an optional /kill webhook (implemented in the orchestrator).
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id
        self._enabled = bool(self.bot_token and self.chat_id)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def send(self, message: str, parse_mode: str = "HTML") -> None:
        """Send a text message to the configured Telegram chat."""
        if not self._enabled:
            return

        import httpx

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    log.warning("telegram_send_failed", status=resp.status_code)
        except Exception as exc:
            log.warning("telegram_send_error", error=str(exc))

    # ── Convenience methods ─────────────────────────────────────────────────
    async def notify_signal(self, market: str, zscore: float, v_ratio: float) -> None:
        await self.send(
            f"🔔 <b>Panic Signal</b>\n"
            f"Market: <code>{market[:60]}</code>\n"
            f"Z-score: {zscore:.2f}  |  Vol ratio: {v_ratio:.1f}×"
        )

    async def notify_entry(
        self, pos_id: str, market: str, price: float, size: float, target: float
    ) -> None:
        await self.send(
            f"📥 <b>Entry Filled</b>\n"
            f"Pos: {pos_id}  |  {market[:40]}\n"
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

    async def notify_kill(self) -> None:
        await self.send("🛑 <b>KILL SWITCH ACTIVATED</b> — all orders cancelled, bot stopping.")
