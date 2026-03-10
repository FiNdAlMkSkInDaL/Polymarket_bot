"""
Shared-memory IPC layer for cross-process L2 order book exchange.

Provides lock-free, sub-microsecond reads of reconstructed order books
from L2 worker processes using a double-buffered Seqlock (Sequence Lock)
architecture.  Each asset gets a fixed-size shared memory block with two
data buffers so readers never block writers and always get a consistent
snapshot.

Components
----------
``SharedBookWriter``
    Used inside L2 worker processes to publish updated book state.
``SharedBookReader``
    Used in the main process to read the latest book state.

Memory layout (per asset, total = 3_424 bytes):
    Control region (16 bytes):
        seqlock_counter uint64      8   (even = idle, odd = write in progress)
        active_block    uint8       1   (0 = Block A active, 1 = Block B active)
        _pad            7 bytes

    Block A — Data block (1_704 bytes), offset 16:
        Header (104 bytes):
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

    Block B — Data block (1_704 bytes), offset 1_720:
        (identical layout to Block A)
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.l2_book import BookState

_log = logging.getLogger(__name__)

# ── Layout constants ───────────────────────────────────────────────────────
MAX_LEVELS = 50
_HEADER_FMT = "<QddddddddBBB5xHHII"  # little-endian
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 104 bytes
_LEVEL_FMT = "<dd"  # price + size
_LEVEL_SIZE = struct.calcsize(_LEVEL_FMT)  # 16 bytes
_LEVELS_BLOCK = MAX_LEVELS * _LEVEL_SIZE  # 800 bytes per side

# ── Seqlock control region ─────────────────────────────────────────────────
_SEQ_FMT = "<Q"                                        # uint64 seqlock counter
_SEQ_SIZE = struct.calcsize(_SEQ_FMT)                  # 8 bytes
_ACTIVE_FMT = "<B"                                     # uint8 active block index
_CONTROL_SIZE = 16                                     # 8 (seq) + 1 (active) + 7 (pad)

_SEQ_OFFSET = 0
_ACTIVE_OFFSET = _SEQ_SIZE                             # 8

# ── Data block (header + bids + asks, NO lock byte) ────────────────────────
_DATA_BLOCK_SIZE = _HEADER_SIZE + 2 * _LEVELS_BLOCK    # 1704 bytes

# ── Block offsets within shared memory ─────────────────────────────────────
_BLOCK_A_OFFSET = _CONTROL_SIZE                        # 16
_BLOCK_B_OFFSET = _CONTROL_SIZE + _DATA_BLOCK_SIZE     # 1720
BLOCK_SIZE = _CONTROL_SIZE + 2 * _DATA_BLOCK_SIZE      # 3424 bytes

# ── Seqlock reader limits ──────────────────────────────────────────────────
_MAX_SEQLOCK_RETRIES = 10

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
    """Writes L2 book state into a double-buffered shared memory segment
    using a Seqlock protocol.

    The writer always writes to the *standby* block (the one readers are
    NOT currently reading), then atomically flips the active-block index.
    The surrounding sequence-counter increments let readers detect any
    concurrent mutation and retry.

    Parameters
    ----------
    asset_id:
        Token ID that this writer publishes.
    shm_name:
        Name of the ``SharedMemory`` segment (created externally).
    """

    __slots__ = ("asset_id", "_shm", "_buf")

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
        """Pack the full book state into the standby block using a Seqlock.

        Protocol:
            1. Increment seqlock counter → odd  (signals write-in-progress)
            2. Write data to the standby block
            3. Toggle active-block index
            4. Increment seqlock counter → even (signals write-complete)
        """
        buf = self._buf

        # ── Step 1: increment seqlock counter to ODD (write-in-progress) ──
        cur_seq = struct.unpack_from(_SEQ_FMT, buf, _SEQ_OFFSET)[0]
        struct.pack_into(_SEQ_FMT, buf, _SEQ_OFFSET, cur_seq + 1)

        # ── Step 2: determine standby block and write into it ─────────────
        active = struct.unpack_from(_ACTIVE_FMT, buf, _ACTIVE_OFFSET)[0] & 1
        standby_idx = 1 - active
        block_off = _BLOCK_A_OFFSET if standby_idx == 0 else _BLOCK_B_OFFSET

        bids_off = block_off + _HEADER_SIZE
        asks_off = block_off + _HEADER_SIZE + _LEVELS_BLOCK

        # Pack header into standby block
        struct.pack_into(
            _HEADER_FMT,
            buf,
            block_off,
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
        n_bids = min(n_bid_levels, MAX_LEVELS)
        offset = bids_off
        for i in range(n_bids):
            struct.pack_into(_LEVEL_FMT, buf, offset, bid_levels[i][0], bid_levels[i][1])
            offset += _LEVEL_SIZE
        for _ in range(n_bids, MAX_LEVELS):
            struct.pack_into(_LEVEL_FMT, buf, offset, 0.0, 0.0)
            offset += _LEVEL_SIZE

        # Pack ask levels
        n_asks = min(n_ask_levels, MAX_LEVELS)
        offset = asks_off
        for i in range(n_asks):
            struct.pack_into(_LEVEL_FMT, buf, offset, ask_levels[i][0], ask_levels[i][1])
            offset += _LEVEL_SIZE
        for _ in range(n_asks, MAX_LEVELS):
            struct.pack_into(_LEVEL_FMT, buf, offset, 0.0, 0.0)
            offset += _LEVEL_SIZE

        # ── Step 3: flip active-block pointer to the freshly-written block ─
        struct.pack_into(_ACTIVE_FMT, buf, _ACTIVE_OFFSET, standby_idx)

        # ── Step 4: increment seqlock counter to EVEN (write-complete) ─────
        struct.pack_into(_SEQ_FMT, buf, _SEQ_OFFSET, cur_seq + 2)

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
    """Reads L2 book state from a double-buffered shared memory segment
    using the Seqlock protocol (lock-free on the reader side).

    Parameters
    ----------
    asset_id:
        Token ID this reader is associated with.
    shm_name:
        Name of the ``SharedMemory`` segment.
    """

    __slots__ = (
        "asset_id", "_shm", "_buf",
        "_last_seq", "_last_snapshot",
    )

    def __init__(self, asset_id: str, shm_name: str) -> None:
        self.asset_id = asset_id
        self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
        self._buf = self._shm.buf
        self._last_seq: int = 0
        self._last_snapshot: SharedBookSnapshot | None = None

    # ── internal: resolve active block offset ──────────────────────────────
    @staticmethod
    def _block_offset(active_idx: int) -> int:
        return _BLOCK_A_OFFSET if (active_idx & 1) == 0 else _BLOCK_B_OFFSET

    # ── public API ─────────────────────────────────────────────────────────
    def read_header(self) -> SharedBookSnapshot:
        """Read only the header fields (no level arrays).

        Fast path for consumers that only need BBO / spread / depth.
        Uses the Seqlock retry protocol — never blocks the writer.
        """
        buf = self._buf

        for _attempt in range(_MAX_SEQLOCK_RETRIES):
            # Step 1-2: read seqlock counter; if odd a write is in-flight
            s1 = struct.unpack_from(_SEQ_FMT, buf, _SEQ_OFFSET)[0]
            if s1 & 1:
                continue  # writer busy → retry immediately

            # Step 3: copy header from the active block into local memory
            active = struct.unpack_from(_ACTIVE_FMT, buf, _ACTIVE_OFFSET)[0]
            block_off = self._block_offset(active)
            header_bytes = bytes(buf[block_off: block_off + _HEADER_SIZE])

            # Step 4: re-read seqlock counter
            s2 = struct.unpack_from(_SEQ_FMT, buf, _SEQ_OFFSET)[0]

            # Step 5: consistency check
            if s1 != s2:
                continue  # concurrent write detected → retry

            # ── success: unpack from the local copy ───────────────────────
            fields = struct.unpack_from(_HEADER_FMT, header_bytes, 0)
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

        # ── all retries exhausted (near-impossible with double buffering) ─
        if self._last_snapshot is not None:
            return self._last_snapshot
        return SharedBookSnapshot(asset_id=self.asset_id)

    def read_full(self) -> SharedBookSnapshot:
        """Read header + all bid/ask levels (lock-free Seqlock)."""
        buf = self._buf

        for _attempt in range(_MAX_SEQLOCK_RETRIES):
            # Step 1-2: read seqlock counter
            s1 = struct.unpack_from(_SEQ_FMT, buf, _SEQ_OFFSET)[0]
            if s1 & 1:
                continue

            # Step 3: snapshot the entire active data block into local memory
            active = struct.unpack_from(_ACTIVE_FMT, buf, _ACTIVE_OFFSET)[0]
            block_off = self._block_offset(active)
            local = bytes(buf[block_off: block_off + _DATA_BLOCK_SIZE])

            # Step 4: re-read seqlock counter
            s2 = struct.unpack_from(_SEQ_FMT, buf, _SEQ_OFFSET)[0]

            # Step 5: consistency check
            if s1 != s2:
                continue

            # ── success: unpack from the local copy ───────────────────────
            fields = struct.unpack_from(_HEADER_FMT, local, 0)
            (
                seq, timestamp, server_time, best_bid, best_ask,
                bid_depth_usd, ask_depth_usd, spread_score, depth_near_mid,
                state, latency_state, is_reliable_byte,
                n_bid, n_ask, delta_count, desync_total,
            ) = fields

            # Bid levels (offsets relative to local copy)
            bids_off = _HEADER_SIZE
            bids: list[tuple[float, float]] = []
            offset = bids_off
            for _ in range(n_bid):
                price, size = struct.unpack_from(_LEVEL_FMT, local, offset)
                if size > 0:
                    bids.append((price, size))
                offset += _LEVEL_SIZE

            # Ask levels
            asks_off = _HEADER_SIZE + _LEVELS_BLOCK
            asks: list[tuple[float, float]] = []
            offset = asks_off
            for _ in range(n_ask):
                price, size = struct.unpack_from(_LEVEL_FMT, local, offset)
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

        # ── all retries exhausted (near-impossible with double buffering) ─
        if self._last_snapshot is not None:
            return self._last_snapshot
        return SharedBookSnapshot(asset_id=self.asset_id)

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
    # Clean up stale segment from a prior crash before creating
    try:
        stale = shared_memory.SharedMemory(name=name, create=False)
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass
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
