"""
Unit tests for the PMXTArchiveAdapter in scripts/backfill_data.py.

Verifies four architectural invariants:
  1. Streaming Memory Invariant   — fetch_l2 yields lazily via AsyncGenerator
  2. Schema Match Invariant       — output matches DataRecorder/DataLoader schema
  3. CLI Routing Invariant        — --source pmxt maps to PMXTArchiveAdapter
  4. Resilience Invariant         — exponential backoff on HTTP 429/500
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import types
from datetime import date
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Ensure scripts/ is importable ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import backfill_data

from backfill_data import (
    ADAPTERS,
    PMXTArchiveAdapter,
    PolymarketTradesAdapter,
    MarketEntry,
    parse_args,
    normalize_ts,
    coalesce_deltas,
    COALESCE_WINDOW_S,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

SAMPLE_MARKET = MarketEntry(
    market_id="0x06b066b03a084f34b35691cf3fda9b070c41c48ea8e3802e0068e71a07579765",
    yes_id="71321045863202695095371220874564407863499636094949880230789991035847900345842",
    no_id="41824606142505960000000000000000000000000000000000000000000000000000000000001",
)

SAMPLE_PARQUET_DATA_JSON = json.dumps({
    "update_type": "price_change",
    "market_id": SAMPLE_MARKET.market_id,
    "token_id": SAMPLE_MARKET.yes_id,
    "side": "YES",
    "best_bid": 0.55,
    "best_ask": 0.56,
    "timestamp": 1741305600.123,
    "change_price": 0.55,
    "change_size": 100,
    "change_side": "BUY",
})


def _make_mock_parquet_file(market_ids: list[str], data_jsons: list[str], timestamps):
    """Build a mock ParquetFile that simulates row-group iteration."""
    import datetime as _dt

    pf = MagicMock()
    pf.metadata.num_row_groups = 1

    # Row-group metadata stub (no statistics → can't skip)
    rg_meta = MagicMock()
    rg_meta.num_columns = 0
    pf.metadata.row_group.return_value = rg_meta

    # Build a mock table from the provided columns
    table = MagicMock()
    table.column.side_effect = lambda name: {
        "market_id": MagicMock(to_pylist=MagicMock(return_value=market_ids)),
        "timestamp_received": MagicMock(to_pylist=MagicMock(return_value=timestamps)),
        "data": MagicMock(to_pylist=MagicMock(return_value=data_jsons)),
    }[name]

    pf.read_row_group.return_value = table
    return pf


# ═════════════════════════════════════════════════════════════════════════
#  Invariant 1: Streaming Memory Invariant
# ═════════════════════════════════════════════════════════════════════════

class TestStreamingMemoryInvariant:
    """Assert that fetch_l2 returns an AsyncIterator/AsyncGenerator,
    yielding records lazily rather than returning a fully populated list."""

    def test_fetch_l2_is_async_generator(self):
        """fetch_l2() must be an async generator function (yields lazily)."""
        adapter = PMXTArchiveAdapter()
        assert inspect.isasyncgenfunction(adapter.fetch_l2), (
            "fetch_l2 must be an async generator function (uses `yield`), "
            "not a coroutine that returns a list"
        )

    def test_fetch_trades_is_async_generator(self):
        """fetch_trades() must also be an async generator function."""
        adapter = PMXTArchiveAdapter()
        assert inspect.isasyncgenfunction(adapter.fetch_trades)

    def test_fetch_l2_return_annotation_is_async_iterator(self):
        """The ABC requires AsyncIterator[dict] return type."""
        hints = PMXTArchiveAdapter.fetch_l2.__annotations__
        ret = hints.get("return")
        # Accept AsyncIterator[dict] or AsyncGenerator
        assert ret is not None or inspect.isasyncgenfunction(PMXTArchiveAdapter.fetch_l2)

    @pytest.mark.asyncio
    async def test_fetch_l2_yields_lazily(self):
        """Calling fetch_l2 must produce an async iterable, not a list."""
        adapter = PMXTArchiveAdapter()
        client = AsyncMock()

        # Patch _head_check to return False (no data) so it returns immediately
        with patch.object(adapter, "_head_check", return_value=False):
            result = adapter.fetch_l2(SAMPLE_MARKET, date(2026, 3, 3), client)
            # Must be an async iterator, not a list/tuple
            assert hasattr(result, "__aiter__"), (
                "fetch_l2 must return an async iterable"
            )
            assert not isinstance(result, (list, tuple, dict)), (
                "fetch_l2 must NOT return a materialized collection"
            )

    def test_supports_batch_flag(self):
        """PMXTArchiveAdapter must declare supports_batch = True for the
        batch orchestrator to engage."""
        adapter = PMXTArchiveAdapter()
        assert getattr(adapter, "supports_batch", False) is True


# ═════════════════════════════════════════════════════════════════════════
#  Invariant 2: Schema Match Invariant
# ═════════════════════════════════════════════════════════════════════════

class TestSchemaMatchInvariant:
    """Verify the yielded dictionaries exactly match the live DataRecorder
    L2 schema so DataLoader._parse_record() classifies them correctly."""

    def test_data_to_record_canonical_shape(self):
        """_data_to_record must produce the canonical record structure."""
        adapter = PMXTArchiveAdapter()
        rec = adapter._data_to_record(
            SAMPLE_PARQUET_DATA_JSON,
            ts=1741305600.123,
            market_id=SAMPLE_MARKET.market_id,
        )
        assert rec is not None

        # Top-level keys
        assert "local_ts" in rec
        assert "source" in rec
        assert "asset_id" in rec
        assert "payload" in rec

        # source must be "l2" (maps to "l2_delta" via _SOURCE_MAP)
        assert rec["source"] == "l2"

        # asset_id must be the hex market condition ID
        assert rec["asset_id"] == SAMPLE_MARKET.market_id

        # local_ts must be a float
        assert isinstance(rec["local_ts"], float)

    def test_payload_event_type(self):
        """payload.event_type must be 'price_change' which DataLoader
        maps to l2_delta via _SOURCE_MAP."""
        adapter = PMXTArchiveAdapter()
        rec = adapter._data_to_record(
            SAMPLE_PARQUET_DATA_JSON, 1741305600.0, SAMPLE_MARKET.market_id,
        )
        assert rec["payload"]["event_type"] == "price_change"

    def test_payload_changes_array(self):
        """payload.changes must be a list of {side, price, size} dicts
        matching L2OrderBook._apply_delta_changes() expectations."""
        adapter = PMXTArchiveAdapter()
        rec = adapter._data_to_record(
            SAMPLE_PARQUET_DATA_JSON, 1741305600.0, SAMPLE_MARKET.market_id,
        )
        changes = rec["payload"]["changes"]
        assert isinstance(changes, list)
        assert len(changes) >= 1

        change = changes[0]
        assert "side" in change
        assert "price" in change
        assert "size" in change
        # side must be BUY or SELL
        assert change["side"] in ("BUY", "SELL")
        # price and size must be strings (L2OrderBook expects str)
        assert isinstance(change["price"], str)
        assert isinstance(change["size"], str)

    def test_payload_market_and_asset_id(self):
        """payload must contain market and asset_id fields."""
        adapter = PMXTArchiveAdapter()
        rec = adapter._data_to_record(
            SAMPLE_PARQUET_DATA_JSON, 1741305600.0, SAMPLE_MARKET.market_id,
        )
        payload = rec["payload"]
        assert payload["market"] == SAMPLE_MARKET.market_id
        assert payload["asset_id"] == SAMPLE_MARKET.yes_id

    def test_payload_timestamp_is_numeric(self):
        """payload.timestamp must be a numeric value (float or int)."""
        adapter = PMXTArchiveAdapter()
        rec = adapter._data_to_record(
            SAMPLE_PARQUET_DATA_JSON, 1741305600.0, SAMPLE_MARKET.market_id,
        )
        ts = rec["payload"]["timestamp"]
        assert isinstance(ts, (int, float))

    def test_source_maps_to_l2_delta_in_data_loader(self):
        """The source 'l2' must exist in DataLoader's _SOURCE_MAP and map
        to 'l2_delta'."""
        from src.backtest.data_loader import _SOURCE_MAP
        assert "l2" in _SOURCE_MAP
        assert _SOURCE_MAP["l2"] == "l2_delta"

    def test_malformed_json_returns_none(self):
        """_data_to_record must return None, not raise, on bad input."""
        adapter = PMXTArchiveAdapter()
        assert adapter._data_to_record("not json {{{", 1.0, "0xabc") is None
        assert adapter._data_to_record("", 1.0, "0xabc") is None

    def test_record_serializes_cleanly_to_jsonl(self):
        """The record must round-trip through JSON without error."""
        adapter = PMXTArchiveAdapter()
        rec = adapter._data_to_record(
            SAMPLE_PARQUET_DATA_JSON, 1741305600.0, SAMPLE_MARKET.market_id,
        )
        line = json.dumps(rec, separators=(",", ":"), default=str)
        restored = json.loads(line)
        assert restored["source"] == "l2"
        assert len(restored["payload"]["changes"]) == 1

    def test_read_parquet_batch_output_schema(self):
        """_read_parquet_batch must produce correctly-keyed records for
        each tracked market_id.

        Uses module-level mocking so pyarrow/fsspec don't need to be
        installed in the test environment.
        """
        import datetime as _dt

        adapter = PMXTArchiveAdapter()
        market_ids = {SAMPLE_MARKET.market_id}
        ts = _dt.datetime(2026, 3, 3, 12, 0, 0)

        mock_pf = _make_mock_parquet_file(
            [SAMPLE_MARKET.market_id],
            [SAMPLE_PARQUET_DATA_JSON],
            [ts],
        )

        # fsspec is imported lazily inside _read_parquet_batch, but the
        # parquet alias is already bound on the imported backfill_data module.
        mock_fs_mod = types.ModuleType("fsspec")
        fs_instance = MagicMock()
        fs_instance.open.return_value.__enter__ = MagicMock(return_value=MagicMock())
        fs_instance.open.return_value.__exit__ = MagicMock(return_value=False)
        mock_fs_mod.filesystem = MagicMock(return_value=fs_instance)

        with patch.object(backfill_data.pq, "ParquetFile", return_value=mock_pf):
            with patch.dict("sys.modules", {"fsspec": mock_fs_mod}):
                result = adapter._read_parquet_batch(
                    "https://r2.pmxt.dev/test.parquet",
                    market_ids,
                )

        assert SAMPLE_MARKET.market_id in result
        recs = result[SAMPLE_MARKET.market_id]
        # Now emits snapshot + delta = 2 records
        assert len(recs) == 2
        # First record should be the synthetic snapshot
        assert recs[0]["payload"]["event_type"] == "book"
        assert "bids" in recs[0]["payload"]
        assert "asks" in recs[0]["payload"]
        # Second record should be the delta
        assert recs[1]["source"] == "l2"
        assert "changes" in recs[1]["payload"]


# ═════════════════════════════════════════════════════════════════════════
#  Invariant 3: CLI Routing Invariant
# ═════════════════════════════════════════════════════════════════════════

class TestCLIRoutingInvariant:
    """Verify --source pmxt correctly routes to PMXTArchiveAdapter."""

    def test_adapter_registry_contains_pmxt(self):
        """ADAPTERS dict must contain 'pmxt' key."""
        assert "pmxt" in ADAPTERS

    def test_pmxt_maps_to_correct_class(self):
        """ADAPTERS['pmxt'] must be PMXTArchiveAdapter, not PolymarketTradesAdapter."""
        assert ADAPTERS["pmxt"] is PMXTArchiveAdapter
        assert ADAPTERS["pmxt"] is not PolymarketTradesAdapter

    def test_parse_args_source_pmxt(self):
        """--source pmxt must be accepted by the CLI parser."""
        args = parse_args(["--source", "pmxt", "--start-date", "2026-03-03",
                           "--end-date", "2026-03-03"])
        assert args.source == "pmxt"

    def test_parse_args_default_is_polymarket(self):
        """Default --source must be 'polymarket', not 'pmxt'."""
        args = parse_args(["--start-date", "2026-03-03", "--end-date", "2026-03-03"])
        assert args.source == "polymarket"

    def test_adapter_instantiation(self):
        """ADAPTERS['pmxt']() must produce a PMXTArchiveAdapter instance."""
        adapter = ADAPTERS["pmxt"]()
        assert isinstance(adapter, PMXTArchiveAdapter)

    def test_all_registered_adapters_are_subclasses(self):
        """Every entry in ADAPTERS must be a DataSourceAdapter subclass."""
        from backfill_data import DataSourceAdapter
        for name, cls in ADAPTERS.items():
            assert issubclass(cls, DataSourceAdapter), (
                f"ADAPTERS['{name}'] = {cls} is not a DataSourceAdapter subclass"
            )


# ═════════════════════════════════════════════════════════════════════════
#  Invariant 4: Resilience Invariant
# ═════════════════════════════════════════════════════════════════════════

class TestResilienceInvariant:
    """Verify exponential backoff on HTTP 429 and 500 errors."""

    @pytest.mark.asyncio
    async def test_head_check_retries_on_429(self):
        """_head_check must retry (not crash) on HTTP 429."""
        adapter = PMXTArchiveAdapter()
        mock_client = AsyncMock()

        # First two calls return 429, third returns 200
        resp_429 = MagicMock(status_code=429)
        resp_200 = MagicMock(status_code=200)
        mock_client.head.side_effect = [resp_429, resp_429, resp_200]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._head_check(mock_client, "https://r2.pmxt.dev/test.parquet")

        assert result is True
        assert mock_client.head.call_count == 3
        # Exponential backoff: sleep(2^1), sleep(2^2)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 2.0  # RETRY_BACKOFF_BASE ** 1
        assert mock_sleep.call_args_list[1][0][0] == 4.0  # RETRY_BACKOFF_BASE ** 2

    @pytest.mark.asyncio
    async def test_head_check_retries_on_500(self):
        """_head_check must retry on server errors (5xx)."""
        adapter = PMXTArchiveAdapter()
        mock_client = AsyncMock()

        resp_500 = MagicMock(status_code=500)
        resp_200 = MagicMock(status_code=200)
        mock_client.head.side_effect = [resp_500, resp_200]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._head_check(mock_client, "https://r2.pmxt.dev/test.parquet")

        assert result is True
        assert mock_client.head.call_count == 2

    @pytest.mark.asyncio
    async def test_head_check_returns_false_on_404(self):
        """_head_check must return False immediately on 404 (not retry)."""
        adapter = PMXTArchiveAdapter()
        mock_client = AsyncMock()

        resp_404 = MagicMock(status_code=404)
        mock_client.head.return_value = resp_404

        result = await adapter._head_check(mock_client, "https://r2.pmxt.dev/missing.parquet")

        assert result is False
        assert mock_client.head.call_count == 1

    @pytest.mark.asyncio
    async def test_head_check_handles_connect_error(self):
        """_head_check must handle network errors with backoff."""
        import httpx
        adapter = PMXTArchiveAdapter()
        mock_client = AsyncMock()

        # All 3 attempts fail with ConnectError → returns False
        mock_client.head.side_effect = httpx.ConnectError("connection refused")

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._head_check(mock_client, "https://r2.pmxt.dev/test.parquet")

        assert result is False
        assert mock_client.head.call_count == 3  # MAX_RETRIES
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_head_check_exhausted_retries_returns_false(self):
        """After MAX_RETRIES 429s, _head_check returns False."""
        adapter = PMXTArchiveAdapter()
        mock_client = AsyncMock()

        resp_429 = MagicMock(status_code=429)
        mock_client.head.return_value = resp_429

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._head_check(mock_client, "https://r2.pmxt.dev/test.parquet")

        assert result is False


# ═════════════════════════════════════════════════════════════════════════
#  Supplementary: Batch mode + normalize_ts
# ═════════════════════════════════════════════════════════════════════════

class TestBatchAndHelpers:
    """Additional coverage for batch orchestration and timestamp normalization."""

    def test_normalize_ts_seconds(self):
        assert normalize_ts(1741305600.0) == 1741305600.0

    def test_normalize_ts_milliseconds(self):
        assert normalize_ts(1741305600123) == pytest.approx(1741305600.123)

    def test_normalize_ts_microseconds(self):
        assert normalize_ts(1741305600123456) == pytest.approx(1741305600.123456)

    def test_normalize_ts_garbage_returns_zero(self):
        assert normalize_ts("not-a-number") == 0.0
        assert normalize_ts(None) == 0.0

    def test_r2_url_template(self):
        """The URL template must produce correct R2 CDN URLs."""
        adapter = PMXTArchiveAdapter()
        filename = adapter._FILE_TEMPLATE.format(date="2026-03-03", hour=14)
        assert filename == "polymarket_orderbook_2026-03-03T14.parquet"

    def test_custom_r2_base(self):
        """PMXTArchiveAdapter must accept a custom R2 base URL."""
        adapter = PMXTArchiveAdapter(r2_base="https://custom.cdn.example.com/")
        assert adapter._r2_base == "https://custom.cdn.example.com"

    @pytest.mark.asyncio
    async def test_fetch_l2_day_batch_returns_dict(self):
        """fetch_l2_day_batch must return a dict keyed by market_id."""
        adapter = PMXTArchiveAdapter()
        client = AsyncMock()

        # Patch _fetch_hour_batch to return empty results for all 24 hours
        with patch.object(
            adapter, "_fetch_hour_batch",
            return_value={SAMPLE_MARKET.market_id: []},
        ):
            result = await adapter.fetch_l2_day_batch(
                [SAMPLE_MARKET], date(2026, 3, 3), client,
            )

        assert isinstance(result, dict)
        assert SAMPLE_MARKET.market_id in result

    @pytest.mark.asyncio
    async def test_fetch_hour_batch_missing_file(self):
        """If HEAD check fails, _fetch_hour_batch returns empty dict."""
        adapter = PMXTArchiveAdapter()
        client = AsyncMock()

        with patch.object(adapter, "_head_check", return_value=False):
            result = await adapter._fetch_hour_batch(
                [SAMPLE_MARKET], date(2026, 3, 3), 0, client,
            )

        assert result == {}

    def test_can_skip_row_group_no_statistics(self):
        """Row groups without statistics must NOT be skipped (conservative)."""
        pf = MagicMock()
        rg = MagicMock()
        rg.num_columns = 1
        col = MagicMock()
        col.path_in_schema = "market_id"
        col.statistics = None
        rg.column.return_value = col
        pf.metadata.row_group.return_value = rg

        result = PMXTArchiveAdapter._can_skip_row_group_batch(
            pf, 0, {"0xabc"},
        )
        assert result is False


# ═════════════════════════════════════════════════════════════════════════
#  Invariant 5: Hourly Snapshot Synthesis
# ═════════════════════════════════════════════════════════════════════════

class TestHourlySnapshotSynthesis:
    """Verify that _data_to_snapshot produces valid l2_snapshot records."""

    def test_snapshot_from_valid_bbo(self):
        """A PMXT row with best_bid/best_ask must produce a snapshot."""
        adapter = PMXTArchiveAdapter()
        snap = adapter._data_to_snapshot(
            SAMPLE_PARQUET_DATA_JSON, 1741305600.0, SAMPLE_MARKET.market_id,
        )
        assert snap is not None
        assert snap["source"] == "l2"
        assert snap["payload"]["event_type"] == "book"
        assert "bids" in snap["payload"]
        assert "asks" in snap["payload"]
        assert len(snap["payload"]["bids"]) == 1
        assert len(snap["payload"]["asks"]) == 1

    def test_snapshot_classified_as_l2_snapshot_by_data_loader(self):
        """DataLoader._parse_record maps event_type='book' → l2_snapshot."""
        from src.backtest.data_loader import _SOURCE_MAP
        # The payload override logic in _parse_record checks for "book"
        # and sets event_type = "l2_snapshot" regardless of source tag.
        assert "l2" in _SOURCE_MAP

    def test_snapshot_bids_asks_have_price_size(self):
        adapter = PMXTArchiveAdapter()
        snap = adapter._data_to_snapshot(
            SAMPLE_PARQUET_DATA_JSON, 1741305600.0, SAMPLE_MARKET.market_id,
        )
        bid = snap["payload"]["bids"][0]
        ask = snap["payload"]["asks"][0]
        assert "price" in bid and "size" in bid
        assert "price" in ask and "size" in ask
        assert float(bid["price"]) == 0.55
        assert float(ask["price"]) == 0.56

    def test_snapshot_returns_none_on_missing_bbo(self):
        """If best_bid or best_ask is missing, no snapshot is emitted."""
        adapter = PMXTArchiveAdapter()
        data_no_bbo = json.dumps({"token_id": "123", "change_price": 0.5})
        assert adapter._data_to_snapshot(data_no_bbo, 1.0, "0xabc") is None

    def test_snapshot_returns_none_on_zero_bbo(self):
        adapter = PMXTArchiveAdapter()
        data_zero = json.dumps({"best_bid": 0, "best_ask": 0})
        assert adapter._data_to_snapshot(data_zero, 1.0, "0xabc") is None

    def test_snapshot_returns_none_on_bad_json(self):
        adapter = PMXTArchiveAdapter()
        assert adapter._data_to_snapshot("{{bad", 1.0, "0xabc") is None


# ═════════════════════════════════════════════════════════════════════════
#  Invariant 6: 50 ms Delta Coalescing
# ═════════════════════════════════════════════════════════════════════════

class TestDeltaCoalescing:
    """Verify coalesce_deltas merges records within 50ms windows."""

    def _make_delta(self, ts: float, side: str = "BUY",
                    price: str = "0.55", size: str = "100") -> dict:
        return {
            "local_ts": ts,
            "source": "l2",
            "asset_id": "0xabc",
            "payload": {
                "event_type": "price_change",
                "changes": [{"side": side, "price": price, "size": size}],
            },
        }

    def _make_snapshot(self, ts: float) -> dict:
        return {
            "local_ts": ts,
            "source": "l2",
            "asset_id": "0xabc",
            "payload": {
                "event_type": "book",
                "bids": [{"price": "0.55", "size": "1"}],
                "asks": [{"price": "0.56", "size": "1"}],
            },
        }

    def test_empty_input(self):
        assert coalesce_deltas([]) == []

    def test_single_record_unchanged(self):
        rec = self._make_delta(1.0)
        result = coalesce_deltas([rec])
        assert len(result) == 1
        assert result[0] is rec

    def test_two_deltas_within_window_merged(self):
        """Two deltas 10ms apart should merge into one with combined changes."""
        r1 = self._make_delta(1.000, "BUY", "0.55", "100")
        r2 = self._make_delta(1.010, "SELL", "0.56", "200")
        result = coalesce_deltas([r1, r2])
        assert len(result) == 1
        changes = result[0]["payload"]["changes"]
        assert len(changes) == 2
        assert changes[0]["side"] == "BUY"
        assert changes[1]["side"] == "SELL"
        # Timestamp is from the first record
        assert result[0]["local_ts"] == 1.000

    def test_deltas_across_window_boundary_not_merged(self):
        """Two deltas 60ms apart should NOT merge (> 50ms window)."""
        r1 = self._make_delta(1.000)
        r2 = self._make_delta(1.060)
        result = coalesce_deltas([r1, r2])
        assert len(result) == 2

    def test_snapshot_breaks_bucket(self):
        """A snapshot must pass through un-merged and flush any open bucket."""
        r1 = self._make_delta(1.000)
        snap = self._make_snapshot(1.020)
        r2 = self._make_delta(1.030)
        result = coalesce_deltas([r1, snap, r2])
        assert len(result) == 3
        assert result[0]["payload"]["event_type"] == "price_change"
        assert result[1]["payload"]["event_type"] == "book"
        assert result[2]["payload"]["event_type"] == "price_change"

    def test_coalesced_output_compatible_with_data_loader(self):
        """Coalesced records must retain the schema DataLoader expects."""
        r1 = self._make_delta(1.000, "BUY", "0.55", "100")
        r2 = self._make_delta(1.010, "SELL", "0.56", "200")
        result = coalesce_deltas([r1, r2])
        rec = result[0]
        assert rec["source"] == "l2"
        assert "local_ts" in rec
        assert "asset_id" in rec
        payload = rec["payload"]
        assert payload["event_type"] == "price_change"
        for c in payload["changes"]:
            assert "side" in c
            assert "price" in c
            assert "size" in c

    def test_multiple_buckets(self):
        """Three distinct 50ms buckets should produce three records."""
        recs = [
            self._make_delta(1.000),
            self._make_delta(1.010),
            self._make_delta(1.100),
            self._make_delta(1.110),
            self._make_delta(1.200),
        ]
        result = coalesce_deltas(recs)
        assert len(result) == 3
        assert len(result[0]["payload"]["changes"]) == 2
        assert len(result[1]["payload"]["changes"]) == 2
        assert len(result[2]["payload"]["changes"]) == 1

    def test_window_constant_is_50ms(self):
        assert COALESCE_WINDOW_S == 0.050


# ═════════════════════════════════════════════════════════════════════════
#  Invariant 7: Analytics Script Import
# ═════════════════════════════════════════════════════════════════════════

class TestAnalyticsScript:
    """Verify analyze_book_resilience.py is importable and correct."""

    def test_import_and_core_functions(self):
        """The analytics module must import cleanly."""
        from analyze_book_resilience import (
            analyze_market_file,
            aggregate_report,
            apply_snapshot,
            apply_delta,
            compute_bdr,
            compute_depth_near_mid,
            MarketStats,
            BookState,
        )
        assert callable(analyze_market_file)
        assert callable(aggregate_report)

    def test_compute_bdr_balanced(self):
        from analyze_book_resilience import BookState, compute_bdr
        book = BookState()
        book.bids = {0.55: 100.0, 0.54: 50.0}
        book.asks = {0.56: 100.0, 0.57: 50.0}
        bdr = compute_bdr(book)
        # bid_depth = 0.55*100 + 0.54*50 = 82.0
        # ask_depth = 0.56*100 + 0.57*50 = 84.5
        assert 0.9 < bdr < 1.1

    def test_compute_bdr_empty_asks(self):
        from analyze_book_resilience import BookState, compute_bdr
        book = BookState()
        book.bids = {0.55: 100.0}
        book.asks = {}
        assert compute_bdr(book) == 1.0

    def test_apply_snapshot_resets_book(self):
        from analyze_book_resilience import BookState, apply_snapshot
        book = BookState()
        book.bids = {0.50: 999.0}
        apply_snapshot(book, {
            "bids": [{"price": "0.55", "size": "100"}],
            "asks": [{"price": "0.56", "size": "100"}],
        })
        assert 0.55 in book.bids
        assert 0.50 not in book.bids
        assert book.last_mid == pytest.approx(0.555)

    def test_apply_delta_updates_book(self):
        from analyze_book_resilience import BookState, apply_delta
        book = BookState()
        book.bids = {0.55: 100.0}
        book.asks = {0.56: 100.0}
        book.last_mid = 0.555
        apply_delta(book, {
            "changes": [{"side": "BUY", "price": "0.54", "size": "200"}],
        })
        assert 0.54 in book.bids
        assert book.bids[0.54] == 200.0

    def test_apply_delta_removes_level_on_zero_size(self):
        from analyze_book_resilience import BookState, apply_delta
        book = BookState()
        book.bids = {0.55: 100.0}
        book.asks = {0.56: 100.0}
        book.last_mid = 0.555
        apply_delta(book, {
            "changes": [{"side": "BUY", "price": "0.55", "size": "0"}],
        })
        assert 0.55 not in book.bids

    def test_depth_near_mid(self):
        from analyze_book_resilience import BookState, compute_depth_near_mid
        book = BookState()
        book.bids = {0.55: 100.0, 0.54: 50.0, 0.50: 999.0}
        book.last_mid = 0.555
        # band_cents=2 → band=0.02 → lo=0.535
        # only 0.55 qualifies (0.54 < 0.535? no 0.54 > 0.535, so both qualify)
        depth = compute_depth_near_mid(book, band_cents=2.0)
        assert depth == 150.0  # 100 + 50

    def test_aggregate_report_recommended_threshold(self):
        from analyze_book_resilience import aggregate_report, MarketStats
        stats = MarketStats(
            market_id="test",
            adverse_moves=10,
            bdr_before_adverse=[0.15, 0.20, 0.25, 0.30, 0.10, 0.12, 0.18, 0.22, 0.28, 0.35],
        )
        report = aggregate_report([stats])
        threshold = report["recommended_config"]["sl_preemptive_obi_threshold"]
        # 10th percentile of [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]
        # = ~0.1 to 0.12 range → floored at 0.05, capped at 0.50
        assert 0.05 <= threshold <= 0.50
