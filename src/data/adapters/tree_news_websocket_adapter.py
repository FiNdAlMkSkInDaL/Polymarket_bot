from __future__ import annotations

from typing import Any

from src.core.config import settings
from src.data.adapters.websocket_adapter_base import WebSocketOracleAdapter
from src.data.oracle_adapter import OracleMarketConfig, OracleSnapshot


class TreeNewsWebSocketAdapter(WebSocketOracleAdapter):
    """Standalone WebSocket adapter for Tree News event-resolution updates."""

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
            or settings.strategy.oracle_tree_news_ws_url
        )
        resolved_api_key = (
            api_key
            or str(oracle_params.get("api_key", ""))
            or settings.strategy.oracle_tree_news_ws_key
        )
        if not resolved_websocket_url:
            raise ValueError("TreeNewsWebSocketAdapter requires ORACLE_TREE_NEWS_WS_URL or oracle_params.websocket_url")

        super().__init__(
            market_config,
            websocket_url=resolved_websocket_url,
            api_key=resolved_api_key,
            **kwargs,
        )
        self._event_id = market_config.external_id
        self._target_outcome = market_config.target_outcome
        self._oracle_params = oracle_params

    @property
    def name(self) -> str:
        return "tree_news_ws"

    def _subscription_messages(self) -> list[str | bytes | dict[str, Any] | list[Any]]:
        payloads = self._oracle_params.get("subscription_payloads")
        if isinstance(payloads, list):
            return payloads

        payload = self._oracle_params.get("subscription_payload")
        if payload is not None:
            return [payload]

        if self._event_id:
            return [{"action": "subscribe", "eventId": self._event_id}]
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
        if message_type in {"subscribed", "ack", "hello"}:
            return None

        event_id = self._extract_event_id(decoded)
        if self._event_id and event_id and event_id != self._event_id:
            return None

        resolved = self._extract_resolved_outcome(decoded)
        confidence = self._extract_confidence(decoded, resolved)
        phase = self._extract_phase(decoded, resolved, confidence)

        if resolved is None and confidence == 0.0 and phase == "pre_event":
            return OracleSnapshot(
                adapter_name=self.name,
                market_id=self._config.market_id,
                raw_state=decoded,
                resolved_outcome=None,
                confidence=0.0,
                event_phase=phase,
            )

        return OracleSnapshot(
            adapter_name=self.name,
            market_id=self._config.market_id,
            raw_state=decoded,
            resolved_outcome=resolved,
            confidence=confidence,
            event_phase=phase,
        )

    def _extract_event_id(self, payload: dict[str, Any]) -> str:
        for key in ("eventId", "event_id", "storyId", "story_id", "id"):
            value = payload.get(key)
            if value:
                return str(value)
        if isinstance(payload.get("event"), dict):
            nested = payload["event"]
            for key in ("id", "eventId", "event_id"):
                value = nested.get(key)
                if value:
                    return str(value)
        return ""

    def _extract_resolved_outcome(self, payload: dict[str, Any]) -> str | None:
        target = self._target_outcome.strip().lower()
        explicit = self._first_string(
            payload,
            "resolved_outcome",
            "resolvedOutcome",
            "winner",
            "calledFor",
            "call",
        )
        if explicit:
            return self._map_outcome_label(explicit, target)

        outcome = payload.get("outcome")
        if isinstance(outcome, dict):
            label = self._first_string(outcome, "label", "name", "value", "winner")
            if label:
                return self._map_outcome_label(label, target)
            binary = outcome.get("resolved")
            if isinstance(binary, bool):
                return "YES" if binary else "NO"
        elif isinstance(outcome, str):
            return self._map_outcome_label(outcome, target)

        signal = payload.get("signal")
        if isinstance(signal, dict):
            label = self._first_string(signal, "direction", "stance", "label")
            if label:
                return self._map_outcome_label(label, target)

        return None

    def _extract_confidence(self, payload: dict[str, Any], resolved: str | None) -> float:
        numeric = self._first_number(
            payload,
            "confidence",
            "confidence_score",
            "confidenceScore",
            "probability",
            "score",
        )
        if numeric is None and isinstance(payload.get("outcome"), dict):
            numeric = self._first_number(
                payload["outcome"],
                "confidence",
                "probability",
                "score",
            )

        if numeric is not None:
            if numeric > 1.0:
                numeric /= 100.0
            return max(0.0, min(1.0, numeric))

        status = self._first_string(payload, "status", "state", "phase").lower()
        urgency = self._first_string(payload, "urgency", "priority", "importance").lower()
        consensus = bool(payload.get("consensus") or payload.get("confirmed") or payload.get("verified"))
        if resolved is None:
            return 0.0
        if consensus and status in {"resolved", "final", "confirmed"}:
            return 0.99
        if status in {"resolved", "final", "confirmed"}:
            return 0.97
        if urgency in {"breaking", "critical"}:
            return 0.90
        if status in {"live", "developing", "active"}:
            return 0.82
        return 0.70

    def _extract_phase(self, payload: dict[str, Any], resolved: str | None, confidence: float) -> str:
        status = self._first_string(payload, "status", "state", "phase").lower()
        urgency = self._first_string(payload, "urgency", "priority", "importance").lower()

        if status in {"resolved", "final", "confirmed", "closed"} or resolved is not None and confidence >= 0.97:
            return "final"
        if urgency in {"breaking", "critical"} or confidence >= 0.90:
            return "critical"
        if status in {"live", "developing", "active", "open"} or resolved is not None:
            return "in_progress"
        if status in {"idle", "paused"}:
            return "idle"
        return "pre_event"

    @staticmethod
    def _first_string(payload: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return str(value)
        return ""

    @staticmethod
    def _first_number(payload: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _map_outcome_label(label: str, target: str) -> str | None:
        normalized = label.strip().lower()
        if normalized in {"yes", "true", "resolved_yes", "buy", "up"}:
            return "YES"
        if normalized in {"no", "false", "resolved_no", "sell", "down"}:
            return "NO"
        if target:
            return "YES" if normalized == target else "NO"
        return None