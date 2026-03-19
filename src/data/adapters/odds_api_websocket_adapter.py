from __future__ import annotations

from typing import Any

from src.core.config import settings
from src.data.adapters.websocket_adapter_base import WebSocketOracleAdapter
from src.data.oracle_adapter import OracleMarketConfig, OracleSnapshot


class OddsAPIWebSocketAdapter(WebSocketOracleAdapter):
    """Standalone WebSocket adapter for live sports odds and score updates."""

    def __init__(
        self,
        market_config: OracleMarketConfig,
        *,
        websocket_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        oracle_params = dict(market_config.oracle_params)
        resolved_websocket_url = (
            websocket_url
            or str(oracle_params.get("websocket_url", ""))
            or str(oracle_params.get("ws_url", ""))
            or settings.strategy.oracle_odds_api_ws_url
        )
        resolved_api_key = (
            api_key
            or str(oracle_params.get("api_key", ""))
            or settings.strategy.oracle_odds_api_ws_key
        )
        if not resolved_websocket_url:
            raise ValueError("OddsAPIWebSocketAdapter requires ORACLE_ODDS_API_WS_URL or oracle_params.websocket_url")

        super().__init__(
            market_config,
            websocket_url=resolved_websocket_url,
            api_key=resolved_api_key,
            **kwargs,
        )
        self._match_id = market_config.external_id
        self._team = market_config.target_outcome
        self._market_type = market_config.market_type or "winner"
        self._goal_line = float(market_config.goal_line)
        self._oracle_params = oracle_params

    @property
    def name(self) -> str:
        return "odds_api_ws"

    def _subscription_messages(self) -> list[str | bytes | dict[str, Any] | list[Any]]:
        payloads = self._oracle_params.get("subscription_payloads")
        if isinstance(payloads, list):
            return payloads

        payload = self._oracle_params.get("subscription_payload")
        if payload is not None:
            return [payload]

        if self._match_id:
            return [{"action": "subscribe", "eventId": self._match_id}]
        return []

    async def _payload_to_snapshot(self, websocket: Any, payload: Any) -> OracleSnapshot | None:
        decoded = self._decode_payload(payload)
        if isinstance(decoded, str):
            lowered = decoded.lower()
            if lowered == "ping":
                await websocket.send("pong")
            return None

        if not isinstance(decoded, dict):
            return None

        message_type = str(decoded.get("type", decoded.get("action", ""))).lower()
        if message_type in {"ping", "heartbeat"}:
            await websocket.send('{"type":"pong"}')
            return None
        if message_type in {"subscribed", "ack", "hello", "snapshot_start", "snapshot_end"}:
            return None

        event = self._extract_event(decoded)
        if event is None:
            return None
        if self._match_id and str(event.get("id", event.get("eventId", ""))) != self._match_id:
            return None

        normalized = {"match": self._normalize_event(event, decoded)}
        return self._build_snapshot(normalized)

    def _extract_event(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        for key in ("event", "match", "data"):
            value = payload.get(key)
            if isinstance(value, dict):
                if "event" in value and isinstance(value["event"], dict):
                    return value["event"]
                return value
        if "home_team" in payload or "homeTeam" in payload:
            return payload
        return None

    def _normalize_event(self, event: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        home_team = str(event.get("home_team", event.get("homeTeam", "")))
        away_team = str(event.get("away_team", event.get("awayTeam", "")))
        status = str(
            event.get("status")
            or event.get("state")
            or event.get("game_status")
            or payload.get("status")
            or "TIMED"
        ).upper()
        completed = bool(event.get("completed", False))
        if completed and status not in {"FINISHED", "AWARDED", "AFTER_EXTRA_TIME", "AFTER_PENALTIES"}:
            status = "FINISHED"

        minute = self._extract_minute(event)
        home_score, away_score = self._extract_scores(event)
        events = event.get("events") if isinstance(event.get("events"), list) else []

        return {
            "status": status,
            "minute": minute,
            "score": {"home": home_score, "away": away_score},
            "homeTeam": {"name": home_team},
            "awayTeam": {"name": away_team},
            "events": events,
        }

    @staticmethod
    def _extract_minute(event: dict[str, Any]) -> int:
        clock = event.get("clock")
        if isinstance(clock, dict):
            for key in ("minute", "minutes", "elapsed"):
                value = clock.get(key)
                if value is not None:
                    return int(float(value))
        for key in ("minute", "minutes", "elapsed", "time_elapsed"):
            value = event.get(key)
            if value is not None:
                return int(float(value))
        return 0

    @staticmethod
    def _extract_scores(event: dict[str, Any]) -> tuple[int, int]:
        if isinstance(event.get("score"), dict):
            score = event["score"]
            return int(score.get("home", 0)), int(score.get("away", 0))

        home_team = str(event.get("home_team", event.get("homeTeam", "")))
        away_team = str(event.get("away_team", event.get("awayTeam", "")))
        home_score = int(event.get("home_score", event.get("homeScore", 0)) or 0)
        away_score = int(event.get("away_score", event.get("awayScore", 0)) or 0)

        if "scores" in event and isinstance(event["scores"], list):
            for row in event["scores"]:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name", row.get("team", "")))
                value = int(float(row.get("score", row.get("points", 0)) or 0))
                if name == home_team:
                    home_score = value
                elif name == away_team:
                    away_score = value

        return home_score, away_score

    def _build_snapshot(self, data: dict[str, Any]) -> OracleSnapshot:
        match = data.get("match", data)
        status = str(match.get("status", "TIMED")).upper()
        minute = int(match.get("minute", 0))
        score = match.get("score", {})
        home_goals = int(score.get("home", 0))
        away_goals = int(score.get("away", 0))
        home_team = str(match.get("homeTeam", {}).get("name", ""))
        away_team = str(match.get("awayTeam", {}).get("name", ""))
        events = match.get("events", [])

        if status in ("FINISHED", "AWARDED", "AFTER_EXTRA_TIME", "AFTER_PENALTIES"):
            phase = "final"
        elif status in ("HALFTIME", "BREAK"):
            phase = "idle"
        elif status in ("IN_PLAY", "LIVE", "EXTRA_TIME"):
            phase = "critical" if minute >= 80 else "in_progress"
        else:
            phase = "pre_event"

        resolved: str | None = None
        confidence = 0.0
        total_goals = home_goals + away_goals
        is_finished = status in ("FINISHED", "AWARDED", "AFTER_EXTRA_TIME", "AFTER_PENALTIES")

        if self._market_type == "winner":
            resolved, confidence = self._resolve_winner(
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                minute=minute,
                is_finished=is_finished,
                events=events,
                phase=phase,
            )
        elif self._market_type == "over_goals":
            resolved, confidence = self._resolve_over_goals(
                total_goals=total_goals,
                is_finished=is_finished,
                phase=phase,
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
        *,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        minute: int,
        is_finished: bool,
        events: list[Any],
        phase: str,
    ) -> tuple[str | None, float]:
        target_lower = self._team.lower()
        if home_team.lower() == target_lower:
            target_goals, opponent_goals = home_goals, away_goals
            opponent_side = "away"
        elif away_team.lower() == target_lower:
            target_goals, opponent_goals = away_goals, home_goals
            opponent_side = "home"
        else:
            return None, 0.0

        goal_diff = target_goals - opponent_goals
        if is_finished:
            return ("YES" if goal_diff > 0 else "NO"), 0.98
        if phase == "pre_event" or goal_diff == 0:
            return None, 0.0

        resolved = "YES" if goal_diff > 0 else "NO"
        opponent_reds = sum(
            1
            for event in events
            if isinstance(event, dict)
            and str(event.get("type", "")).upper() == "RED_CARD"
            and str(event.get("team", "")).lower() == opponent_side
        )
        if abs(goal_diff) >= 2 and minute >= 80 and opponent_reds > 0:
            return resolved, 0.92
        if abs(goal_diff) >= 2 and minute >= 80:
            return resolved, 0.90
        return resolved, 0.85

    def _resolve_over_goals(
        self,
        *,
        total_goals: int,
        is_finished: bool,
        phase: str,
    ) -> tuple[str | None, float]:
        if is_finished:
            return ("YES" if total_goals > self._goal_line else "NO"), 0.98
        if phase == "pre_event":
            return None, 0.0
        if total_goals > self._goal_line:
            return "YES", 0.85
        return None, 0.0