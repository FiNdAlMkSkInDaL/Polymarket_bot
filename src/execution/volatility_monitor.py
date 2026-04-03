from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VolatilityStatus:
    timestamp_ms: int
    window_ms: int
    asset_id: str | None
    sigma_cents: float | None
    sample_count: int
    max_safe_volatility_cents: float
    is_breached: bool


class RollingMidPriceVolatilityMonitor:
    def __init__(self, *, window_ms: int, max_safe_volatility_cents: float) -> None:
        if not isinstance(window_ms, int) or window_ms <= 0:
            raise ValueError("window_ms must be a strictly positive int")
        if not math.isfinite(max_safe_volatility_cents) or max_safe_volatility_cents < 0.0:
            raise ValueError("max_safe_volatility_cents must be a finite float >= 0")
        self._window_ms = window_ms
        self._max_safe_volatility_cents = float(max_safe_volatility_cents)
        self._samples: dict[str, deque[tuple[int, float]]] = {}
        self._sum_cents: dict[str, float] = {}
        self._sum_sq_cents: dict[str, float] = {}

    def record_mid_price(
        self,
        asset_id: str,
        mid_price: float,
        timestamp_ms: int,
    ) -> VolatilityStatus:
        asset_key = str(asset_id or "").strip()
        if not asset_key:
            return self.current_status(timestamp_ms)
        if not math.isfinite(mid_price) or mid_price <= 0.0:
            return self.current_status(timestamp_ms)

        ts_ms = int(timestamp_ms)
        mid_cents = float(mid_price) * 100.0
        samples = self._samples.setdefault(asset_key, deque())
        samples.append((ts_ms, mid_cents))
        self._sum_cents[asset_key] = self._sum_cents.get(asset_key, 0.0) + mid_cents
        self._sum_sq_cents[asset_key] = self._sum_sq_cents.get(asset_key, 0.0) + (mid_cents * mid_cents)
        self._prune_asset(asset_key, ts_ms)
        return self.current_status(ts_ms)

    def current_status(self, timestamp_ms: int) -> VolatilityStatus:
        ts_ms = int(timestamp_ms)
        best_asset_id: str | None = None
        best_sigma_cents: float | None = None
        best_sample_count = 0

        for asset_id in list(self._samples):
            self._prune_asset(asset_id, ts_ms)
            samples = self._samples.get(asset_id)
            if not samples:
                continue
            sigma_cents = self._sigma_cents(asset_id)
            if sigma_cents is None:
                continue
            if best_sigma_cents is None or sigma_cents > best_sigma_cents:
                best_asset_id = asset_id
                best_sigma_cents = sigma_cents
                best_sample_count = len(samples)

        is_breached = (
            self._max_safe_volatility_cents > 0.0
            and best_sigma_cents is not None
            and best_sigma_cents > self._max_safe_volatility_cents
        )
        return VolatilityStatus(
            timestamp_ms=ts_ms,
            window_ms=self._window_ms,
            asset_id=best_asset_id,
            sigma_cents=best_sigma_cents,
            sample_count=best_sample_count,
            max_safe_volatility_cents=self._max_safe_volatility_cents,
            is_breached=is_breached,
        )

    def _prune_asset(self, asset_id: str, timestamp_ms: int) -> None:
        samples = self._samples.get(asset_id)
        if not samples:
            return
        cutoff_ms = int(timestamp_ms) - self._window_ms
        while samples and samples[0][0] < cutoff_ms:
            _, old_cents = samples.popleft()
            self._sum_cents[asset_id] = self._sum_cents.get(asset_id, 0.0) - old_cents
            self._sum_sq_cents[asset_id] = self._sum_sq_cents.get(asset_id, 0.0) - (old_cents * old_cents)
        if not samples:
            self._samples.pop(asset_id, None)
            self._sum_cents.pop(asset_id, None)
            self._sum_sq_cents.pop(asset_id, None)

    def _sigma_cents(self, asset_id: str) -> float | None:
        samples = self._samples.get(asset_id)
        if not samples:
            return None
        sample_count = len(samples)
        if sample_count == 1:
            return 0.0
        total = self._sum_cents.get(asset_id, 0.0)
        total_sq = self._sum_sq_cents.get(asset_id, 0.0)
        mean = total / sample_count
        variance = max(0.0, (total_sq / sample_count) - (mean * mean))
        return math.sqrt(variance)