from __future__ import annotations

from threading import Lock


class ClobNonceManager:
    def __init__(self, starting_nonce: int = 0) -> None:
        if not isinstance(starting_nonce, int) or starting_nonce < 0:
            raise ValueError("starting_nonce must be a non-negative int")
        self._lock = Lock()
        self._next_nonce = starting_nonce

    def reserve_nonces(self, count: int = 1) -> tuple[int, ...]:
        if not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive int")
        with self._lock:
            start = self._next_nonce
            self._next_nonce += count
            return tuple(range(start, self._next_nonce))

    def sync_nonce(self, venue_nonce: int) -> None:
        if not isinstance(venue_nonce, int) or venue_nonce < 0:
            raise ValueError("venue_nonce must be a non-negative int")
        with self._lock:
            self._next_nonce = max(self._next_nonce, venue_nonce + 1)

    @property
    def next_nonce(self) -> int:
        with self._lock:
            return self._next_nonce