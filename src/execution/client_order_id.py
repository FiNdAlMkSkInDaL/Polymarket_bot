from __future__ import annotations

from typing import Literal


ClientOrderSignalSource = Literal["OFI", "SI9", "CTF", "CONTAGION", "MANUAL", "REWARD"]


def _require_non_empty_string(name: str, value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


class ClientOrderIdGenerator:
    def __init__(
        self,
        signal_source: ClientOrderSignalSource,
        session_id: str,
    ) -> None:
        normalized_signal_source = str(signal_source or "").strip().upper()
        if normalized_signal_source not in {"OFI", "SI9", "CTF", "CONTAGION", "MANUAL", "REWARD"}:
            raise ValueError(f"Unsupported signal_source: {signal_source!r}")

        normalized_session_id = _require_non_empty_string("session_id", session_id)

        self._signal_source = normalized_signal_source
        self._session_id = normalized_session_id
        self._session_prefix = normalized_session_id[:8]

    @property
    def signal_source(self) -> ClientOrderSignalSource:
        return self._signal_source

    def for_signal_source(self, signal_source: ClientOrderSignalSource) -> "ClientOrderIdGenerator":
        return ClientOrderIdGenerator(signal_source, self._session_id)

    def generate(
        self,
        market_id: str,
        side: Literal["YES", "NO"],
        timestamp_ms: int,
    ) -> str:
        normalized_market_id = _require_non_empty_string("market_id", market_id)
        if side not in {"YES", "NO"}:
            raise ValueError(f"Unsupported side: {side!r}")
        if not isinstance(timestamp_ms, int):
            raise ValueError("timestamp_ms must be an int")

        client_order_id = (
            f"{self._signal_source}-{self._session_prefix}-{normalized_market_id[:8]}-{side[0]}-{timestamp_ms}"
        )
        if len(client_order_id) >= 64:
            raise ValueError("client_order_id must be shorter than 64 characters")
        return client_order_id