"""
Shared data types used across the orderbook and L2 book modules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Level:
    """A single price level in the book."""

    price: float
    size: float
