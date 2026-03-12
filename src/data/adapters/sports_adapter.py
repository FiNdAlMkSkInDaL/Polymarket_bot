"""
Sports Adapter — polls live sports APIs for match state and results.

Translates match status, goals, and events into graduated confidence
scores and event phases for the SI-8 Oracle Latency Arbitrage path.

Confidence derivation
─────────────────────
  ┌─────────────────────────────────────────┬────────────┐
  │ Condition                               │ Confidence │
  ├─────────────────────────────────────────┼────────────┤
  │ Goal scored, match still live           │    0.85    │
  │ Red card + 2-goal lead, > 80 min       │    0.92    │
  │ Final whistle (FT / AET)               │    0.98    │
  └─────────────────────────────────────────┴────────────┘

Event phases
────────────
  ``"pre_event"`` — before kickoff (5 s interval)
  ``"in_progress"`` — match live (1 s interval)
  ``"critical"`` — last 10 minutes or stoppage time (200 ms interval)
  ``"idle"`` — halftime / break (30 s interval)
  ``"final"`` — full-time (5 s interval)
"""

from __future__ import annotations

from typing import Any

import aiohttp

from src.core.config import settings
from src.core.logger import get_logger
from src.data.oracle_adapter import OffChainOracleAdapter, OracleMarketConfig, OracleSnapshot

log = get_logger(__name__)

_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)


class SportsAdapter(OffChainOracleAdapter):
    """Polls a sports API for live match state.

    Expects ``oracle_params`` in the market config to contain::

        {
            "match_id": "<API match identifier>",
            "team": "<team name whose win maps to YES>",
            "market_type": "winner"   // "winner" or "over_goals"
            "goal_line": 2.5          // only for market_type == "over_goals"
        }
    """

    def __init__(
        self,
        market_config: OracleMarketConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(market_config, **kwargs)
        self._match_id: str = market_config.oracle_params.get("match_id", "")
        self._team: str = market_config.oracle_params.get("team", "")
        self._market_type: str = market_config.oracle_params.get("market_type", "winner")
        self._goal_line: float = float(market_config.oracle_params.get("goal_line", 2.5))
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "sports"

    # ── Dynamic polling interval per event phase ───────────────────────

    def _compute_interval(self, phase: str) -> int:
        _INTERVALS = {
            "pre_event": 5_000,
            "in_progress": 1_000,
            "critical": settings.strategy.oracle_critical_poll_ms,
            "idle": settings.strategy.oracle_idle_poll_ms,
            "final": 5_000,
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
        api_url = settings.strategy.oracle_sports_api_url
        api_key = settings.strategy.oracle_sports_api_key

        url = f"{api_url.rstrip('/')}/matches/{self._match_id}"
        headers = {"X-Auth-Token": api_key} if api_key else {}

        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return self._parse_response(data)

    def _parse_response(self, data: dict) -> OracleSnapshot:
        """Translate raw sports API JSON into an ``OracleSnapshot``.

        Expected response shape (simplified football-data.org style)::

            {
                "match": {
                    "status": "IN_PLAY" | "HALFTIME" | "FINISHED" | "TIMED" | ...,
                    "minute": 78,
                    "score": {
                        "home": 2,
                        "away": 1
                    },
                    "homeTeam": {"name": "TeamA"},
                    "awayTeam": {"name": "TeamB"},
                    "events": [
                        {"type": "RED_CARD", "team": "away", "minute": 65},
                        ...
                    ]
                }
            }
        """
        match = data.get("match", data)
        status = str(match.get("status", "TIMED")).upper()
        minute = int(match.get("minute", 0))
        score = match.get("score", {})
        home_goals = int(score.get("home", 0))
        away_goals = int(score.get("away", 0))
        home_team = str(match.get("homeTeam", {}).get("name", ""))
        away_team = str(match.get("awayTeam", {}).get("name", ""))
        events = match.get("events", [])

        # ── Determine event phase ──────────────────────────────────
        if status in ("FINISHED", "AWARDED", "AFTER_EXTRA_TIME", "AFTER_PENALTIES"):
            phase = "final"
        elif status in ("HALFTIME", "BREAK"):
            phase = "idle"
        elif status in ("IN_PLAY", "LIVE", "EXTRA_TIME"):
            phase = "critical" if minute >= 80 else "in_progress"
        else:
            phase = "pre_event"

        # ── Resolve outcome based on market_type ───────────────────
        resolved: str | None = None
        confidence = 0.0

        total_goals = home_goals + away_goals
        is_finished = status in ("FINISHED", "AWARDED", "AFTER_EXTRA_TIME", "AFTER_PENALTIES")

        if self._market_type == "winner":
            resolved, confidence = self._resolve_winner(
                home_team, away_team, home_goals, away_goals,
                minute, is_finished, events, phase,
            )
        elif self._market_type == "over_goals":
            resolved, confidence = self._resolve_over_goals(
                total_goals, is_finished, phase,
            )

        return OracleSnapshot(
            adapter_name=self.name,
            market_id=self._config.market_id,
            raw_state=data,
            resolved_outcome=resolved,
            confidence=confidence,
            event_phase=phase,
        )

    def _resolve_winner(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        minute: int,
        is_finished: bool,
        events: list,
        phase: str,
    ) -> tuple[str | None, float]:
        """Determine outcome and confidence for a 'Will team X win?' market."""
        target_lower = self._team.lower()

        # Determine which side the target team is on
        if home_team.lower() == target_lower:
            target_goals, opponent_goals = home_goals, away_goals
        elif away_team.lower() == target_lower:
            target_goals, opponent_goals = away_goals, home_goals
        else:
            # Team not found — cannot resolve
            return None, 0.0

        goal_diff = target_goals - opponent_goals

        if is_finished:
            resolved = "YES" if goal_diff > 0 else "NO"
            return resolved, 0.98

        if phase == "pre_event":
            return None, 0.0

        # Match is in progress
        if goal_diff == 0:
            return None, 0.0  # drawn — no clear outcome

        resolved = "YES" if goal_diff > 0 else "NO"

        # Check for opponent red cards (strengthens confidence)
        opponent_side = "away" if home_team.lower() == target_lower else "home"
        opponent_reds = sum(
            1 for e in events
            if str(e.get("type", "")).upper() == "RED_CARD"
            and str(e.get("team", "")).lower() == opponent_side
        )

        if abs(goal_diff) >= 2 and minute >= 80 and opponent_reds > 0:
            return resolved, 0.92
        elif abs(goal_diff) >= 2 and minute >= 80:
            return resolved, 0.90
        else:
            return resolved, 0.85

    def _resolve_over_goals(
        self,
        total_goals: int,
        is_finished: bool,
        phase: str,
    ) -> tuple[str | None, float]:
        """Determine outcome for an 'Over X.5 goals' market."""
        if is_finished:
            resolved = "YES" if total_goals > self._goal_line else "NO"
            return resolved, 0.98

        if phase == "pre_event":
            return None, 0.0

        # Match in progress — can only resolve YES early (goals already over)
        if total_goals > self._goal_line:
            return "YES", 0.85

        return None, 0.0
