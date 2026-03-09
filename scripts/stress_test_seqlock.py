#!/usr/bin/env python3
"""
Stress-test for the double-buffered Seqlock IPC.

Three verification tasks:
  1. CORRECTNESS  — writer at ~100k writes/sec, reader as fast as possible.
                    Detects "torn reads" (fields from different write epochs).
  2. PERFORMANCE  — measures read_header() and read_full() latency (p50/p99/max).
  3. RESILIENCE   — simulates SIGKILL on writer mid-write; reader must not hang.

Usage (Linux):
    python -m scripts.stress_test_seqlock
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import struct
import statistics
import sys
import time
from multiprocessing import shared_memory

# ── ensure project root is importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.ipc import (
    BLOCK_SIZE,
    SharedBookWriter,
    SharedBookReader,
    _SEQ_FMT,
    _SEQ_OFFSET,
)

ASSET_ID = "stress_test_asset_0xdead"
SHM_NAME = "pmb_stress_test"
N_WRITES = 200_000
N_LEVELS = 10


# ═══════════════════════════════════════════════════════════════════════════
#  Writer process
# ═══════════════════════════════════════════════════════════════════════════
def _writer_proc(
    shm_name: str,
    n_writes: int,
    ready_event,
    done_event,
    kill_mid_write: bool = False,
    kill_after: int = 0,
) -> None:
    """Write n_writes updates as fast as possible.

    Every write sets *all* float fields to the same epoch counter so the
    reader can verify consistency (no field should come from a different
    epoch).
    """
    writer = SharedBookWriter(ASSET_ID, shm_name)
    ready_event.set()

    for epoch in range(1, n_writes + 1):
        val = float(epoch)
        levels = [(val, val)] * N_LEVELS
        writer.write(
            seq=epoch,
            timestamp=val,
            server_time=val,
            best_bid=val,
            best_ask=val,
            bid_depth_usd=val,
            ask_depth_usd=val,
            spread_score=val,
            depth_near_mid=val,
            state=2,
            latency_state=0,
            is_reliable=True,
            n_bid_levels=N_LEVELS,
            n_ask_levels=N_LEVELS,
            delta_count=epoch,
            desync_total=0,
            bid_levels=levels,
            ask_levels=levels,
        )

        if kill_mid_write and epoch == kill_after:
            # Simulate crash by leaving seqlock counter ODD
            buf = writer._buf
            cur_seq = struct.unpack_from(_SEQ_FMT, buf, _SEQ_OFFSET)[0]
            struct.pack_into(_SEQ_FMT, buf, _SEQ_OFFSET, cur_seq + 1)
            # Hard exit without releasing
            os._exit(137)

    writer.close()
    done_event.set()


# ═══════════════════════════════════════════════════════════════════════════
#  Reader process
# ═══════════════════════════════════════════════════════════════════════════
def _reader_proc(
    shm_name: str,
    ready_event,
    done_event,
    result_dict: dict,
) -> None:
    """Read as fast as possible until the writer is done.

    Checks every snapshot for consistency: all float fields must share the
    same epoch value (no torn reads).
    """
    ready_event.wait()

    reader = SharedBookReader(ASSET_ID, shm_name)

    torn_reads = 0
    total_reads = 0
    header_latencies: list[float] = []
    full_latencies: list[float] = []

    # Wait for the first successful read (writer may not have completed
    # its first write yet when we start)
    while not done_event.is_set():
        try:
            reader.read_header()
            break
        except Exception:
            time.sleep(0.0001)

    while not done_event.is_set() or total_reads < 1000:
        # ── header-only read ──────────────────────────────────────────────
        t0 = time.perf_counter_ns()
        snap_h = reader.read_header()
        dt_h = time.perf_counter_ns() - t0
        header_latencies.append(dt_h)
        total_reads += 1

        # All float header fields must be from the same epoch
        epoch_vals = {
            snap_h.timestamp,
            snap_h.server_time,
            snap_h.best_bid,
            snap_h.best_ask,
            snap_h.bid_depth_usd,
            snap_h.ask_depth_usd,
            snap_h.spread_score,
            snap_h.depth_near_mid,
        }
        if len(epoch_vals) > 1:
            torn_reads += 1

        # ── full read (with levels) ───────────────────────────────────────
        t0 = time.perf_counter_ns()
        snap_f = reader.read_full()
        dt_f = time.perf_counter_ns() - t0
        full_latencies.append(dt_f)
        total_reads += 1

        all_vals = {
            snap_f.timestamp,
            snap_f.server_time,
            snap_f.best_bid,
            snap_f.best_ask,
            snap_f.bid_depth_usd,
            snap_f.ask_depth_usd,
            snap_f.spread_score,
            snap_f.depth_near_mid,
        }
        if snap_f.bid_levels:
            for p, s in snap_f.bid_levels:
                all_vals.add(p)
                all_vals.add(s)
        if snap_f.ask_levels:
            for p, s in snap_f.ask_levels:
                all_vals.add(p)
                all_vals.add(s)
        if len(all_vals) > 1:
            torn_reads += 1

    reader.close()

    result_dict["torn_reads"] = torn_reads
    result_dict["total_reads"] = total_reads
    result_dict["header_p50_ns"] = int(statistics.median(header_latencies)) if header_latencies else 0
    result_dict["header_p99_ns"] = int(sorted(header_latencies)[int(len(header_latencies) * 0.99)]) if header_latencies else 0
    result_dict["header_max_ns"] = max(header_latencies) if header_latencies else 0
    result_dict["full_p50_ns"] = int(statistics.median(full_latencies)) if full_latencies else 0
    result_dict["full_p99_ns"] = int(sorted(full_latencies)[int(len(full_latencies) * 0.99)]) if full_latencies else 0
    result_dict["full_max_ns"] = max(full_latencies) if full_latencies else 0
    result_dict["contention_warnings"] = reader._contention_warnings


# ═══════════════════════════════════════════════════════════════════════════
#  Test runners
# ═══════════════════════════════════════════════════════════════════════════
def _alloc_shm() -> shared_memory.SharedMemory:
    try:
        old = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        old.close()
        old.unlink()
    except FileNotFoundError:
        pass
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=BLOCK_SIZE)
    shm.buf[:BLOCK_SIZE] = b"\x00" * BLOCK_SIZE
    return shm


def test_correctness_and_performance() -> bool:
    """Test 1 + Test 2: correctness (no torn reads) + latency stats."""
    print("=" * 70)
    print("TEST 1+2: Correctness & Performance")
    print(f"  Writer: {N_WRITES:,} writes, Reader: as-fast-as-possible")
    print("=" * 70)

    shm = _alloc_shm()
    ready = multiprocessing.Event()
    done = multiprocessing.Event()
    mgr = multiprocessing.Manager()
    results = mgr.dict()

    reader_p = multiprocessing.Process(
        target=_reader_proc, args=(SHM_NAME, ready, done, results),
    )
    writer_p = multiprocessing.Process(
        target=_writer_proc, args=(SHM_NAME, N_WRITES, ready, done),
    )

    reader_p.start()
    writer_p.start()
    writer_p.join(timeout=60)
    done.set()
    reader_p.join(timeout=10)

    torn = results.get("torn_reads", -1)
    total = results.get("total_reads", 0)
    contention = results.get("contention_warnings", 0)

    print(f"\n  Total reads:          {total:>10,}")
    print(f"  Torn reads:           {torn:>10}")
    print(f"  Contention warnings:  {contention:>10}")
    print(f"\n  read_header() latency:")
    print(f"    p50:  {results.get('header_p50_ns', 0):>8,} ns")
    print(f"    p99:  {results.get('header_p99_ns', 0):>8,} ns")
    print(f"    max:  {results.get('header_max_ns', 0):>8,} ns")
    print(f"\n  read_full() latency:")
    print(f"    p50:  {results.get('full_p50_ns', 0):>8,} ns")
    print(f"    p99:  {results.get('full_p99_ns', 0):>8,} ns")
    print(f"    max:  {results.get('full_max_ns', 0):>8,} ns")

    ok = torn == 0
    print(f"\n  RESULT: {'PASS ✓' if ok else 'FAIL ✗ — TORN READS DETECTED'}\n")

    shm.close()
    shm.unlink()
    return ok


def test_process_resilience() -> bool:
    """Test 3: writer SIGKILL mid-write → reader must not hang."""
    print("=" * 70)
    print("TEST 3: Process Resilience (writer crash mid-write)")
    print("=" * 70)

    shm = _alloc_shm()
    ready = multiprocessing.Event()
    done = multiprocessing.Event()

    # Start writer that will crash after 500 writes (leaving seqlock odd)
    writer_p = multiprocessing.Process(
        target=_writer_proc,
        args=(SHM_NAME, 10_000, ready, done),
        kwargs={"kill_mid_write": True, "kill_after": 500},
    )
    writer_p.start()
    ready.wait(timeout=5)
    writer_p.join(timeout=10)

    # Now the seqlock counter should be ODD (writer crashed mid-write)
    seq_val = struct.unpack_from(_SEQ_FMT, shm.buf, _SEQ_OFFSET)[0]
    seq_is_odd = bool(seq_val & 1)
    print(f"\n  Seqlock counter after crash: {seq_val} ({'odd ✓' if seq_is_odd else 'even — crash sim may not have worked'})")

    # Reader must NOT hang — it should fall back gracefully
    reader = SharedBookReader(ASSET_ID, SHM_NAME)

    t0 = time.monotonic()
    try:
        # First read with no cached snapshot returns empty SharedBookSnapshot
        snap = reader.read_header()
        hung = False
        got_data = snap.seq > 0
    except Exception as e:
        hung = False
        got_data = False
        print(f"  Reader raised (unexpected): {type(e).__name__}: {e}")

    elapsed = time.monotonic() - t0
    print(f"  Reader returned in {elapsed*1000:.2f} ms (did not hang)")

    # Second attempt: seed a last-good snapshot and verify fallback
    reader._last_snapshot = SharedBookSnapshot(asset_id=ASSET_ID, seq=42, best_bid=0.55, best_ask=0.60)
    snap2 = reader.read_header()
    fallback_ok = snap2.seq == 42 and snap2.best_bid == 0.55
    print(f"  Fallback to last-good snapshot: {'PASS ✓' if fallback_ok else 'FAIL ✗'}")

    reader.close()
    ok = not hung and elapsed < 1.0
    print(f"\n  RESULT: {'PASS ✓' if ok else 'FAIL ✗ — READER HUNG'}\n")

    shm.close()
    shm.unlink()
    return ok


# Need the import for test_process_resilience fallback test
from src.core.ipc import SharedBookSnapshot


def main() -> None:
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║     SEQLOCK IPC STRESS TEST SUITE                              ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    results = {}
    results["correctness"] = test_correctness_and_performance()
    results["resilience"] = test_process_resilience()

    print("=" * 70)
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {name:20s}: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    print(f"  OVERALL: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
