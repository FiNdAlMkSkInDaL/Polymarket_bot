"""
Arbitrage Cluster Manager — groups mutually exclusive Polymarket
outcomes (negRisk events) into tradeable clusters for SI-9.

Polymarket encodes multi-outcome events (e.g., "Who will win?") as
multiple binary markets sharing the same ``event_id`` with
``negRisk=True``.  Because exactly one outcome resolves YES, the
true probability sum must equal 1.0.  When the sum of YES best-bids
drops below ``1.0 - margin``, a passive maker arbitrage exists.

This module identifies those clusters from the discovered market list
and exposes them to the combo-arb detector signal.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from src.core.config import settings
from src.core.logger import get_logger
from src.data.market_discovery import MarketInfo

log = get_logger(__name__)


@dataclass
class ArbCluster:
    """A group of mutually exclusive markets sharing an event_id."""

    event_id: str
    legs: list[MarketInfo] = field(default_factory=list)
    last_scan_ts: float = 0.0
    active: bool = True

    @property
    def n_legs(self) -> int:
        return len(self.legs)


class ArbitrageClusterManager:
    """Discovers and maintains mutually exclusive event clusters.

    Call :meth:`scan_clusters` after each market discovery cycle to
    rebuild the cluster map.  Only events where *all* sub-markets
    have ``neg_risk=True`` (Polymarket's mutually-exclusive flag),
    and the leg count is between 2 and ``si9_max_legs`` inclusive,
    are admitted.
    """

    def __init__(self) -> None:
        self._clusters: dict[str, ArbCluster] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def scan_clusters(self, markets: list[MarketInfo]) -> list[ArbCluster]:
        """Rebuild the cluster map from the current market list.

        Returns the list of newly-active clusters (for logging / alerting).
        """
        strat = settings.strategy
        max_legs = strat.si9_max_legs

        # Group by event_id
        by_event: dict[str, list[MarketInfo]] = defaultdict(list)
        for m in markets:
            if m.neg_risk and m.event_id:
                by_event[m.event_id].append(m)

        now = time.monotonic()
        new_clusters: list[ArbCluster] = []
        seen_events: set[str] = set()

        for event_id, legs in by_event.items():
            seen_events.add(event_id)
            if len(legs) < 2 or len(legs) > max_legs:
                # Remove clusters that no longer qualify
                self._clusters.pop(event_id, None)
                continue

            existing = self._clusters.get(event_id)
            if existing is None:
                cluster = ArbCluster(
                    event_id=event_id,
                    legs=legs,
                    last_scan_ts=now,
                    active=True,
                )
                self._clusters[event_id] = cluster
                new_clusters.append(cluster)
                log.info(
                    "arb_cluster_discovered",
                    event_id=event_id,
                    n_legs=len(legs),
                    questions=[lg.question[:60] for lg in legs],
                )
            else:
                # Refresh legs (market data may have changed)
                existing.legs = legs
                existing.last_scan_ts = now
                existing.active = True

        # Deactivate clusters whose event vanished from discovery
        for event_id in list(self._clusters):
            if event_id not in seen_events:
                self._clusters[event_id].active = False

        return new_clusters

    @property
    def active_clusters(self) -> list[ArbCluster]:
        """Return all currently active clusters."""
        return [c for c in self._clusters.values() if c.active]

    def get_cluster(self, event_id: str) -> ArbCluster | None:
        c = self._clusters.get(event_id)
        return c if c and c.active else None

    def all_cluster_yes_token_ids(self) -> set[str]:
        """Return YES token IDs across all active clusters (for book subscription)."""
        ids: set[str] = set()
        for c in self.active_clusters:
            for leg in c.legs:
                ids.add(leg.yes_token_id)
        return ids
