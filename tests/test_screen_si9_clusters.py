from __future__ import annotations

import json

import httpx

from scripts import screen_si9_clusters


def test_extract_cluster_filters_to_active_neg_risk_legs() -> None:
    event = {
        "id": "evt-1",
        "slug": "harvey-weinstein-prison-time",
        "title": "Harvey Weinstein prison time?",
        "active": True,
        "closed": False,
        "enableNegRisk": True,
        "markets": [
            {
                "id": "544092",
                "question": "No prison time?",
                "conditionId": "0xf398b0e5016eeaee9b0885ed84012b6dc91269ac10d3b59d60722859c2e30b2f",
                "outcomes": '["Yes", "No"]',
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "enableOrderBook": True,
                "negRisk": True,
                "groupItemTitle": "No Prison Time",
                "volume24hrClob": 2883.265132,
                "liquidityClob": 640.19758,
            },
            {
                "id": "544093",
                "question": "Less than 5 years?",
                "conditionId": "0xe2b48e3b44de9658ee9c8b37354301763e33c0b502fd966839d644b4c0a9dea8",
                "outcomes": '["Yes", "No"]',
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "enableOrderBook": True,
                "negRisk": True,
                "groupItemTitle": "<5 years",
                "volume24hrClob": 2916.127042,
                "liquidityClob": 648.82543,
            },
            {
                "id": "bad-closed",
                "question": "Closed leg",
                "conditionId": "0xdeadbeef",
                "outcomes": '["Yes", "No"]',
                "active": True,
                "closed": True,
                "acceptingOrders": False,
                "enableOrderBook": True,
                "negRisk": True,
                "groupItemTitle": "Closed",
                "volume24hrClob": 1,
                "liquidityClob": 1,
            },
        ],
    }

    cluster = screen_si9_clusters._extract_cluster(event, min_legs=2, max_legs=6)

    assert cluster is not None
    assert cluster.event_id == "evt-1"
    assert cluster.leg_count == 2
    assert [leg.condition_id for leg in cluster.legs] == [
        "0xe2b48e3b44de9658ee9c8b37354301763e33c0b502fd966839d644b4c0a9dea8",
        "0xf398b0e5016eeaee9b0885ed84012b6dc91269ac10d3b59d60722859c2e30b2f",
    ]


def test_extract_cluster_rejects_non_neg_risk_event() -> None:
    event = {
        "id": "evt-2",
        "title": "Not a negRisk cluster",
        "active": True,
        "closed": False,
        "enableNegRisk": False,
        "markets": [],
    }

    cluster = screen_si9_clusters._extract_cluster(event, min_legs=2, max_legs=6)

    assert cluster is None


def test_export_clusters_uses_condition_id_hex_strings(tmp_path) -> None:
    cluster = screen_si9_clusters.RankedCluster(
        event_id="evt-3",
        event_slug="sample-event",
        title="Sample Event",
        leg_count=2,
        total_volume_24h=1000.0,
        total_liquidity=250.0,
        cluster_score=4.0,
        legs=[
            screen_si9_clusters.ClusterLeg(
                market_id="111",
                condition_id="0xabc123",
                question="Leg A",
                group_item_title="A",
                volume_24h=500.0,
                liquidity=100.0,
            ),
            screen_si9_clusters.ClusterLeg(
                market_id="222",
                condition_id="0xdef456",
                question="Leg B",
                group_item_title="B",
                volume_24h=500.0,
                liquidity=150.0,
            ),
        ],
    )
    output_path = tmp_path / "si9_clusters.json"

    screen_si9_clusters.export_clusters([cluster], top_n=1, output_path=output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == [
        {
            "event_id": "evt-3",
            "event_slug": "sample-event",
            "title": "Sample Event",
            "leg_count": 2,
            "total_volume_24h": 1000.0,
            "total_liquidity": 250.0,
            "cluster_score": 4.0,
            "condition_ids": ["0xabc123", "0xdef456"],
            "markets": [
                {
                    "market_id": "111",
                    "condition_id": "0xabc123",
                    "question": "Leg A",
                    "group_item_title": "A",
                    "volume_24h": 500.0,
                    "liquidity": 100.0,
                },
                {
                    "market_id": "222",
                    "condition_id": "0xdef456",
                    "question": "Leg B",
                    "group_item_title": "B",
                    "volume_24h": 500.0,
                    "liquidity": 150.0,
                },
            ],
        }
    ]


def test_fetch_clusters_deduplicates_events(monkeypatch) -> None:
    event = {
        "id": "evt-1",
        "title": "Dedup cluster",
        "slug": "dedup-cluster",
        "active": True,
        "closed": False,
        "enableNegRisk": True,
        "markets": [
            {
                "id": "m1",
                "question": "Leg 1",
                "conditionId": "0x111",
                "outcomes": '["Yes", "No"]',
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "enableOrderBook": True,
                "negRisk": True,
                "volume24hrClob": 10,
                "liquidityClob": 5,
            },
            {
                "id": "m2",
                "question": "Leg 2",
                "conditionId": "0x222",
                "outcomes": '["Yes", "No"]',
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "enableOrderBook": True,
                "negRisk": True,
                "volume24hrClob": 20,
                "liquidityClob": 5,
            },
        ],
    }
    pages = [[event], [event]]

    class DummyResponse:
        def __init__(self, payload: list[dict]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict]:
            return self._payload

    class DummyClient:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = 0

        def __enter__(self) -> DummyClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str, params: dict[str, object]) -> DummyResponse:
            payload = pages[min(self.calls, len(pages) - 1)]
            self.calls += 1
            return DummyResponse(payload)

    monkeypatch.setattr(screen_si9_clusters.httpx, "Client", DummyClient)

    clusters = screen_si9_clusters.fetch_clusters(
        page_size=1,
        max_pages=2,
        timeout=5.0,
        min_legs=2,
        max_legs=6,
    )

    assert len(clusters) == 1
    assert clusters[0].event_id == "evt-1"


def test_get_with_retries_retries_http_status_errors(monkeypatch) -> None:
    attempts: list[int] = []

    class DummyClient:
        def get(self, url: str, params: dict[str, object]) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 3:
                request = httpx.Request("GET", url, params=params)
                response = httpx.Response(status_code=503, request=request)
                raise httpx.HTTPStatusError("boom", request=request, response=response)
            return httpx.Response(status_code=200, json=[], request=httpx.Request("GET", url, params=params))

    monkeypatch.setattr(screen_si9_clusters.time, "sleep", lambda *_args, **_kwargs: None)

    response = screen_si9_clusters._get_with_retries(DummyClient(), "https://example.com", {}, attempts=3)

    assert response.status_code == 200
    assert len(attempts) == 3