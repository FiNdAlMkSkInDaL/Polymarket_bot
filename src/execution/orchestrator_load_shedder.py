from __future__ import annotations

from typing import Any, Iterable, Literal


class OrchestratorLoadShedder:
    def __init__(
        self,
        max_active_l2_markets: int,
        ranked_market_ids: list[str],
        position_manager: Any | None = None,
        deployment_phase: Literal["PAPER", "DRY_RUN", "LIVE"] = "LIVE",
    ) -> None:
        if not isinstance(max_active_l2_markets, int) or max_active_l2_markets <= 0:
            raise ValueError("max_active_l2_markets must be a strictly positive int")
        self._deployment_phase = deployment_phase
        self._position_manager = position_manager
        self._max_active_l2_markets = min(max_active_l2_markets, 25) if deployment_phase == "LIVE" else max_active_l2_markets
        self._allowed_market_ids = self._build_allowed_market_ids(ranked_market_ids)

    @property
    def max_active_l2_markets(self) -> int:
        return self._max_active_l2_markets

    @property
    def allowed_market_ids(self) -> frozenset[str]:
        return self._allowed_market_ids

    def is_market_allowed(self, market_id: str) -> bool:
        market_key = str(market_id or "").strip()
        return bool(market_key) and market_key in self._allowed_market_ids

    def update_target_map(self, ranked_market_ids: list[str]) -> None:
        self._allowed_market_ids = self._build_allowed_market_ids(ranked_market_ids)

    def _build_allowed_market_ids(self, ranked_market_ids: list[str]) -> frozenset[str]:
        top_tier_market_ids = self._top_tier_market_ids(ranked_market_ids)
        retained_exposures = self._current_exposure_market_ids()
        return frozenset((*top_tier_market_ids, *retained_exposures))

    def _top_tier_market_ids(self, ranked_market_ids: list[str]) -> tuple[str, ...]:
        unique_market_ids: list[str] = []
        seen: set[str] = set()
        for market_id in ranked_market_ids:
            market_key = str(market_id or "").strip()
            if not market_key or market_key in seen:
                continue
            seen.add(market_key)
            unique_market_ids.append(market_key)
            if len(unique_market_ids) >= self._max_active_l2_markets:
                break
        return tuple(unique_market_ids)

    def _current_exposure_market_ids(self) -> tuple[str, ...]:
        if self._position_manager is None:
            return tuple()

        open_market_ids = getattr(self._position_manager, "get_open_market_ids", None)
        if callable(open_market_ids):
            try:
                market_ids = tuple(sorted(str(market_id).strip() for market_id in open_market_ids() if str(market_id).strip()))
                if market_ids:
                    return market_ids
            except Exception:
                pass

        open_positions = getattr(self._position_manager, "get_open_positions", None)
        if not callable(open_positions):
            return tuple()

        try:
            positions: Iterable[Any] = open_positions()
        except Exception:
            return tuple()

        market_ids = {
            str(getattr(position, "market_id", "") or "").strip()
            for position in positions
            if str(getattr(position, "market_id", "") or "").strip()
        }
        return tuple(sorted(market_ids))