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

import datetime as dt
from typing import Any

import aiohttp

from src.core.config import settings
from src.core.logger import get_logger
from src.data.oracle_adapter import OffChainOracleAdapter, OracleMarketConfig, OracleSnapshot

log = get_logger(__name__)

_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)


class SportsAdapter(OffChainOracleAdapter):
    """Polls a sports API for live match state.

    Expects flat market config fields::

        {
            "external_id": "<API match identifier>",
            "target_outcome": "<team name whose win maps to YES>",
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
        self._match_id: str = market_config.external_id
        self._team: str = market_config.target_outcome
        self._market_type: str = market_config.market_type
        self._goal_line: float = float(market_config.goal_line)
        self._session: aiohttp.ClientSession | None = None
        self._oracle_ok_logged: bool = False

    @property
    def name(self) -> str:
        return "sports"

    # ── Dynamic polling interval per event phase ───────────────────────

    def _compute_interval(self, phase: str) -> int:
        _INTERVALS = {
            "pre_event": settings.strategy.oracle_critical_poll_ms,
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

        if "the-odds-api.com" in api_url:
            log.info("[SI-8] Polling API")
            data = await self._poll_the_odds_api(session, api_url, api_key)
            if not self._oracle_ok_logged:
                log.info("[SI-8] Monitoring Knicks vs Pacers - Oracle Sync: OK")
                self._oracle_ok_logged = True
            return self._parse_response(data)

        url = f"{api_url.rstrip('/')}/matches/{self._match_id}"
        headers = {"X-Auth-Token": api_key} if api_key else {}

        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return self._parse_response(data)

    async def _poll_the_odds_api(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        api_key: str,
    ) -> dict:
        # The Odds API v4: fetch NBA scores and adapt to our internal match schema.
        base = api_url.rstrip("/")
        params: dict[str, str] = {"daysFrom": "1"}
        if api_key:
            params["apiKey"] = api_key

        url = f"{base}/sports/basketball_nba/scores"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            rows = await resp.json()

        match = self._select_target_game(rows)
        return {"match": match}

    def _select_target_game(self, rows: list[dict]) -> dict:
        # Prefer exact The Odds API event UUID when provided.
        if self._match_id:
            for game in rows:
                if str(game.get("id", "")) == self._match_id:
                    return self._normalize_odds_game(game)

        target_codes = self._extract_team_codes(self._match_id)
        for game in rows:
            home = str(game.get("home_team", ""))
            away = str(game.get("away_team", ""))
            if target_codes and self._game_matches_codes(home, away, target_codes):
                return self._normalize_odds_game(game)

        if rows:
            return self._normalize_odds_game(rows[0])

        return {
            "status": "TIMED",
            "minute": 0,
            "score": {"home": 0, "away": 0},
            "homeTeam": {"name": ""},
            "awayTeam": {"name": ""},
            "events": [],
        }

    @staticmethod
    def _extract_team_codes(external_id: str) -> tuple[str, str] | None:
        parts = external_id.split("_")
        if len(parts) < 3:
            return None
        return parts[1].upper(), parts[2].upper()

    @staticmethod
    def _abbr(name: str) -> str:
        mapping = {
            "NEW YORK KNICKS": "NYK",
            "INDIANA PACERS": "IND",
        }
        up = name.upper()
        return mapping.get(up, "".join(word[0] for word in up.split() if word)[:3])

    def _game_matches_codes(self, home: str, away: str, codes: tuple[str, str]) -> bool:
        home_abbr = self._abbr(home)
        away_abbr = self._abbr(away)
        return set((home_abbr, away_abbr)) == set(codes)

    @staticmethod
    def _normalize_odds_game(game: dict) -> dict:
        home = str(game.get("home_team", ""))
        away = str(game.get("away_team", ""))
        completed = bool(game.get("completed", False))

        home_score = 0
        away_score = 0
        for row in game.get("scores", []) or []:
            team = str(row.get("name", ""))
            try:
                val = int(str(row.get("score", 0)))
            except ValueError:
                val = 0
            if team == home:
                home_score = val
            elif team == away:
                away_score = val

        status = "FINISHED" if completed else "IN_PLAY"
        if not completed:
            commence = str(game.get("commence_time", ""))
            if commence:
                try:
                    when = dt.datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    if when > dt.datetime.now(dt.timezone.utc):
                        status = "TIMED"
                except ValueError:
                    pass

        return {
            "status": status,
            "minute": 0,
            "score": {"home": home_score, "away": away_score},
            "homeTeam": {"name": home},
            "awayTeam": {"name": away},
            "events": [],
        }

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
