"""
Shared-memory IPC layer for cross-process L2 order book exchange.

Provides zero-copy, sub-microsecond reads of reconstructed order books
from L2 worker processes.  Each asset gets a fixed-size shared memory
block laid out as a packed C struct.

Components
----------
``SharedBookWriter``
    Used inside L2 worker processes to publish updated book state.
``SharedBookReader``
    Used in the main process to read the latest book state.

Memory layout (per asset, total ≈ 1_768 bytes):
    Header  (56 bytes):
        seq             uint64      8
        timestamp       float64     8
        server_time     float64     8
        best_bid        float64     8
        best_ask        float64     8
        bid_depth_usd   float64     8
        ask_depth_usd   float64     8
        spread_score    float64     8
        depth_near_mid  float64     8   (pre-computed for ASG fast-path)
        state           uint8       1
        latency_state   uint8       1   (0=HEALTHY 1=DEGRADED 2=BLOCKED)
        is_reliable     uint8       1
        _pad            5 bytes
        n_bid_levels    uint16      2
        n_ask_levels    uint16      2
        delta_count     uint32      4
        desync_total    uint32      4
    Bid levels (50 × 16 = 800 bytes):
        price           float64     8
        size            float64     8
    Ask levels (50 × 16 = 800 bytes):
        price           float64     8
        size            float64     8
    Write lock (8 bytes):
        lock_flag       uint64      8   (0 = free, 1 = writing)
"""

from __future__ import annotations

import ctypes
import logging
import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.l2_book import BookState

_log = logging.getLogger(__name__)


class IPCReadError(Exception):
    """Raised when a shared-memory read cannot return consistent data."""

# ── Layout constants ───────────────────────────────────────────────────────
MAX_LEVELS = 50
_HEADER_FMT = "<QddddddddBBB5xHHII"  # little-endian
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 104 bytes
_LEVEL_FMT = "<dd"  # price + size
_LEVEL_SIZE = struct.calcsize(_LEVEL_FMT)  # 16 bytes
_LEVELS_BLOCK = MAX_LEVELS * _LEVEL_SIZE  # 800 bytes per side
_LOCK_FMT = "<Q"
_LOCK_SIZE = struct.calcsize(_LOCK_FMT)  # 8 bytes
BLOCK_SIZE = _HEADER_SIZE + 2 * _LEVELS_BLOCK + _LOCK_SIZE  # 1712 bytes

_LOCK_OFFSET = _HEADER_SIZE + 2 * _LEVELS_BLOCK
_BIDS_OFFSET = _HEADER_SIZE
_ASKS_OFFSET = _HEADER_SIZE + _LEVELS_BLOCK

# BookState enum value mapping (mirrors l2_book.BookState)
_STATE_MAP = {"empty": 0, "buffering": 1, "synced": 2, "desynced": 3}
_STATE_REVERSE = {0: "empty", 1: "buffering", 2: "synced", 3: "desynced"}

# LatencyState enum value mapping
LATENCY_HEALTHY = 0
LATENCY_DEGRADED = 1
LATENCY_BLOCKED = 2


def _shm_name(asset_id: str) -> str:
    """Derive a valid shared-memory segment name from an asset id.

    Shared memory names must be short and filesystem-safe.  We hash the
    (potentially hex) asset id to produce a compact, safe name.
    """
    import hashlib

    digest = hashlib.sha1(asset_id.encode()).hexdigest()[:16]
    return f"pmb_{digest}"


# ── Compact snapshot returned by readers ───────────────────────────────────
@dataclass(slots=True)
class SharedBookSnapshot:
    """Lightweight snapshot read from shared memory."""

    asset_id: str
    seq: int = 0
    timestamp: float = 0.0
    server_time: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    spread_score: float = 0.0
    depth_near_mid: float = 0.0
    state: int = 0  # raw enum value
    latency_state: int = 0
    is_reliable: bool = True
    n_bid_levels: int = 0
    n_ask_levels: int = 0
    delta_count: int = 0
    desync_total: int = 0
    bid_levels: list[tuple[float, float]] | None = None
    ask_levels: list[tuple[float, float]] | None = None

    @property
    def spread(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return round(self.best_ask - self.best_bid, 4)
        return 0.0

    @property
    def mid_price(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return round((self.best_bid + self.best_ask) / 2.0, 4)
        return 0.0

    @property
    def fresh(self) -> bool:
        return self.latency_state != LATENCY_BLOCKED

    @property
    def state_name(self) -> str:
        return _STATE_REVERSE.get(self.state, "empty")


# ═══════════════════════════════════════════════════════════════════════════
#  Writer (used inside L2 worker processes)
# ═══════════════════════════════════════════════════════════════════════════
class SharedBookWriter:
    """Writes L2 book state into a shared memory segment.

    Parameters
    ----------
    asset_id:
        Token ID that this writer publishes.
    shm_name:
        Name of the ``SharedMemory`` segment (created externally).
    """

    def __init__(self, asset_id: str, shm_name: str) -> None:
        self.asset_id = asset_id
        self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
        self._buf = self._shm.buf

    def write(
        self,
        *,
        seq: int,
        timestamp: float,
        server_time: float,
        best_bid: float,
        best_ask: float,
        bid_depth_usd: float,
        ask_depth_usd: float,
        spread_score: float,
        depth_near_mid: float,
        state: int,
        latency_state: int,
        is_reliable: bool,
        n_bid_levels: int,
        n_ask_levels: int,
        delta_count: int,
        desync_total: int,
        bid_levels: list[tuple[float, float]],
        ask_levels: list[tuple[float, float]],
    ) -> None:
        """Pack the full book state into shared memory.

        The write-lock byte prevents the reader from seeing a torn write.
        """
        buf = self._buf

        # Acquire write lock
        struct.pack_into(_LOCK_FMT, buf, _LOCK_OFFSET, 1)

        # Pack header
        struct.pack_into(
            _HEADER_FMT,
            buf,
            0,
            seq,
            timestamp,
            server_time,
            best_bid,
            best_ask,
            bid_depth_usd,
            ask_depth_usd,
            spread_score,
            depth_near_mid,
            state,
            latency_state,
            1 if is_reliable else 0,
            min(n_bid_levels, MAX_LEVELS),
            min(n_ask_levels, MAX_LEVELS),
            delta_count,
            desync_total,
        )

        # Pack bid levels
        offset = _BIDS_OFFSET
        for i in range(min(n_bid_levels, MAX_LEVELS)):
            struct.pack_into(_LEVEL_FMT, buf, offset, bid_levels[i][0], bid_levels[i][1])
            offset += _LEVEL_SIZE
        # Zero remaining bid slots
        for i in range(min(n_bid_levels, MAX_LEVELS), MAX_LEVELS):
            struct.pack_into(_LEVEL_FMT, buf, offset, 0.0, 0.0)
            offset += _LEVEL_SIZE

        # Pack ask levels
        offset = _ASKS_OFFSET
        for i in range(min(n_ask_levels, MAX_LEVELS)):
            struct.pack_into(_LEVEL_FMT, buf, offset, ask_levels[i][0], ask_levels[i][1])
            offset += _LEVEL_SIZE
        # Zero remaining ask slots
        for i in range(min(n_ask_levels, MAX_LEVELS), MAX_LEVELS):
            struct.pack_into(_LEVEL_FMT, buf, offset, 0.0, 0.0)
            offset += _LEVEL_SIZE

        # Release write lock
        struct.pack_into(_LOCK_FMT, buf, _LOCK_OFFSET, 0)

    def close(self) -> None:
        """Detach from shared memory (does not unlink)."""
        try:
            self._shm.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Reader (used in main process)
# ═══════════════════════════════════════════════════════════════════════════
class SharedBookReader:
    """Reads L2 book state from a shared memory segment.

    Parameters
    ----------
    asset_id:
        Token ID this reader is associated with.
    shm_name:
        Name of the ``SharedMemory`` segment.
    """

    def __init__(self, asset_id: str, shm_name: str) -> None:
        self.asset_id = asset_id
        self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
        self._buf = self._shm.buf
        # Cached last-read seq for change detection
        self._last_seq: int = 0
        # Spin-lock safety: cache last-good snapshot and track failures
        self._last_snapshot: SharedBookSnapshot | None = None
        self._spin_failures: int = 0

    def read_header(self) -> SharedBookSnapshot:
        """Read only the header fields (no level arrays).

        Fast path for consumers that only need BBO / spread / depth.
        Spins briefly if a write is in progress (the lock is held for
        only a few microseconds).
        """
        buf = self._buf
        # Spin on write lock (extremely brief)
        acquired = False
        for _ in range(100):
            lock_val = struct.unpack_from(_LOCK_FMT, buf, _LOCK_OFFSET)[0]
            if lock_val == 0:
                acquired = True
                break

        if not acquired:
            self._spin_failures += 1
            if self._spin_failures % 100 == 1:
                _log.warning(
                    "ipc_spin_lock_timeout",
                    extra={"asset_id": self.asset_id, "total_failures": self._spin_failures},
                )
            if self._last_snapshot is not None:
                return self._last_snapshot
            raise IPCReadError(
                f"Spin-lock timeout on first read for {self.asset_id}"
            )

        # Read header
        fields = struct.unpack_from(_HEADER_FMT, buf, 0)
        (
            seq, timestamp, server_time, best_bid, best_ask,
            bid_depth_usd, ask_depth_usd, spread_score, depth_near_mid,
            state, latency_state, is_reliable_byte,
            n_bid, n_ask, delta_count, desync_total,
        ) = fields

        self._last_seq = seq
        snap = SharedBookSnapshot(
            asset_id=self.asset_id,
            seq=seq,
            timestamp=timestamp,
            server_time=server_time,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth_usd=bid_depth_usd,
            ask_depth_usd=ask_depth_usd,
            spread_score=spread_score,
            depth_near_mid=depth_near_mid,
            state=state,
            latency_state=latency_state,
            is_reliable=bool(is_reliable_byte),
            n_bid_levels=n_bid,
            n_ask_levels=n_ask,
            delta_count=delta_count,
            desync_total=desync_total,
        )
        self._last_snapshot = snap
        return snap

    def read_full(self) -> SharedBookSnapshot:
        """Read header + all bid/ask levels."""
        buf = self._buf
        # Spin on write lock
        acquired = False
        for _ in range(100):
            lock_val = struct.unpack_from(_LOCK_FMT, buf, _LOCK_OFFSET)[0]
            if lock_val == 0:
                acquired = True
                break

        if not acquired:
            self._spin_failures += 1
            if self._spin_failures % 100 == 1:
                _log.warning(
                    "ipc_spin_lock_timeout",
                    extra={"asset_id": self.asset_id, "total_failures": self._spin_failures},
                )
            if self._last_snapshot is not None:
                return self._last_snapshot
            raise IPCReadError(
                f"Spin-lock timeout on first read for {self.asset_id}"
            )

        # Header
        fields = struct.unpack_from(_HEADER_FMT, buf, 0)
        (
            seq, timestamp, server_time, best_bid, best_ask,
            bid_depth_usd, ask_depth_usd, spread_score, depth_near_mid,
            state, latency_state, is_reliable_byte,
            n_bid, n_ask, delta_count, desync_total,
        ) = fields

        # Bid levels
        bids: list[tuple[float, float]] = []
        offset = _BIDS_OFFSET
        for _ in range(n_bid):
            price, size = struct.unpack_from(_LEVEL_FMT, buf, offset)
            if size > 0:
                bids.append((price, size))
            offset += _LEVEL_SIZE

        # Ask levels
        asks: list[tuple[float, float]] = []
        offset = _ASKS_OFFSET
        for _ in range(n_ask):
            price, size = struct.unpack_from(_LEVEL_FMT, buf, offset)
            if size > 0:
                asks.append((price, size))
            offset += _LEVEL_SIZE

        self._last_seq = seq
        snap = SharedBookSnapshot(
            asset_id=self.asset_id,
            seq=seq,
            timestamp=timestamp,
            server_time=server_time,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth_usd=bid_depth_usd,
            ask_depth_usd=ask_depth_usd,
            spread_score=spread_score,
            depth_near_mid=depth_near_mid,
            state=state,
            latency_state=latency_state,
            is_reliable=bool(is_reliable_byte),
            n_bid_levels=n_bid,
            n_ask_levels=n_ask,
            delta_count=delta_count,
            desync_total=desync_total,
            bid_levels=bids,
            ask_levels=asks,
        )
        self._last_snapshot = snap
        return snap

    @property
    def last_seq(self) -> int:
        return self._last_seq

    def close(self) -> None:
        """Detach from shared memory (does not unlink)."""
        try:
            self._shm.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Allocation helpers (used by ProcessManager)
# ═══════════════════════════════════════════════════════════════════════════
def allocate_shm(asset_id: str) -> tuple[shared_memory.SharedMemory, str]:
    """Allocate a shared memory block for one asset.

    Returns the SharedMemory object and its name.  The caller owns the
    lifecycle (must call ``shm.close()`` + ``shm.unlink()``).
    """
    name = _shm_name(asset_id)
    shm = shared_memory.SharedMemory(name=name, create=True, size=BLOCK_SIZE)
    # Zero-initialize
    shm.buf[:BLOCK_SIZE] = b"\x00" * BLOCK_SIZE
    return shm, name


def cleanup_shm(shm: shared_memory.SharedMemory) -> None:
    """Close and unlink a shared memory segment safely."""
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass
