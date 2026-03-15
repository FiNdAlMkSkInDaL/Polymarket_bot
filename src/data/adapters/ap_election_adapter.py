"""
AP Election Adapter — polls Associated Press election race-call feeds.

Translates AP race status metadata into graduated confidence scores and
event phases that drive the SI-8 Oracle Latency Arbitrage fast-strike path.

Confidence derivation
─────────────────────
Rather than hardcoding a dangerous 1.0, confidence is derived from
reporting percentage and call status:

  ┌─────────────────────────────────┬────────────┐
  │ Condition                       │ Confidence │
  ├─────────────────────────────────┼────────────┤
  │ reporting < 50%, projected      │    0.70    │
  │ reporting ≥ 50%, consistent lead│    0.82    │
  │ reporting ≥ 90%, clear lead     │    0.90    │
  │ AP official race call           │    0.97    │
  │ Multi-desk consensus            │    0.99    │
  └─────────────────────────────────┴────────────┘

Event phases
────────────
  ``"pre_event"``   — polls not yet closed (5 s interval)
  ``"in_progress"`` — counting underway (2 s interval)
  ``"critical"``    — > 80% reporting or imminent call (200 ms interval)
  ``"final"``       — race called (1 s interval)
"""

from __future__ import annotations

from typing import Any

import aiohttp

from src.core.config import settings
from src.core.logger import get_logger
from src.data.oracle_adapter import OffChainOracleAdapter, OracleMarketConfig, OracleSnapshot

log = get_logger(__name__)

# Timeout for HTTP requests to the AP API.
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)


class APElectionAdapter(OffChainOracleAdapter):
    """Polls the AP Election API for race calls and vote counts.

    Expects ``oracle_params`` in the market config to contain::

        {
            "race_id": "<AP race identifier>",
            "target_candidate": "<candidate name whose win maps to YES>"
        }
    """

    def __init__(
        self,
        market_config: OracleMarketConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(market_config, **kwargs)
        self._race_id: str = market_config.oracle_params.get("race_id", "")
        self._target: str = market_config.oracle_params.get("target_candidate", "")
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "ap_election"

    # ── Dynamic polling interval per event phase ───────────────────────

    def _compute_interval(self, phase: str) -> int:
        _INTERVALS = {
            "pre_event": 5_000,
            "in_progress": 2_000,
            "critical": settings.strategy.oracle_critical_poll_ms,
            "final": 1_000,
            "idle": settings.strategy.oracle_idle_poll_ms,
        }
        return _INTERVALS.get(phase, settings.strategy.oracle_default_poll_ms)

    # ── HTTP session lifecycle ─────────────────────────────────────────

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT)
        return self._session

    # ── Core poll implementation ───────────────────────────────────────

    async def poll(self) -> OracleSnapshot:
        session = await self._ensure_session()
        api_url = settings.strategy.oracle_ap_api_url
        api_key = settings.strategy.oracle_ap_api_key

        headers = {"x-api-key": api_key} if api_key else {}
        params = {"raceID": self._race_id}

        async with session.get(api_url, headers=headers, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return self._parse_response(data)

    def _parse_response(self, data: dict) -> OracleSnapshot:
        """Translate raw AP API JSON into an ``OracleSnapshot``.

        Expected response shape (simplified)::

            {
                "raceID": "...",
                "reportingPct": 0.85,
                "called": true,
                "calledFor": "CandidateName",
                "candidates": [
                    {"name": "CandidateA", "votes": 1234567},
                    {"name": "CandidateB", "votes": 987654},
                ],
                "status": "counting" | "called" | "pre_event",
                "multiDeskConsensus": false,
            }
        """
        reporting_pct = float(data.get("reportingPct", 0.0))
        called = bool(data.get("called", False))
        called_for = str(data.get("calledFor", ""))
        multi_desk = bool(data.get("multiDeskConsensus", False))
        status = str(data.get("status", "pre_event"))

        # ── Determine event phase ──────────────────────────────────
        if called:
            phase = "final"
        elif reporting_pct >= 0.80:
            phase = "critical"
        elif reporting_pct > 0:
            phase = "in_progress"
        else:
            phase = "pre_event" if status != "idle" else "idle"

        # ── Determine resolved outcome ─────────────────────────────
        resolved: str | None = None
        if called:
            target_lower = self._target.lower()
            resolved = "YES" if called_for.lower() == target_lower else "NO"
        elif reporting_pct > 0 and self._leading_candidate(data):
            # Not yet called but we can derive a preliminary outcome
            leader = self._leading_candidate(data)
            if leader and leader.lower() == self._target.lower():
                resolved = "YES"
            elif leader:
                resolved = "NO"

        # ── Graduated confidence derivation ────────────────────────
        confidence = 0.0
        if multi_desk and called:
            confidence = 0.99
        elif called:
            confidence = 0.97
        elif reporting_pct >= 0.90 and resolved is not None:
            confidence = 0.90
        elif reporting_pct >= 0.50 and resolved is not None:
            confidence = 0.82
        elif reporting_pct > 0 and resolved is not None:
            confidence = 0.70

        return OracleSnapshot(
            adapter_name=self.name,
            market_id=self._config.market_id,
            raw_state=data,
            resolved_outcome=resolved,
            confidence=confidence,
            event_phase=phase,
        )

    @staticmethod
    def _leading_candidate(data: dict) -> str | None:
        """Return the name of the candidate with the most votes, or None."""
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        best = max(candidates, key=lambda c: int(c.get("votes", 0)))
        return str(best.get("name", ""))
