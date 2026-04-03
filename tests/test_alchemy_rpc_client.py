from __future__ import annotations

import os
from typing import Any

import pytest

from src.data.alchemy_rpc_client import AlchemyRpcClient


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeHttpClient:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.is_closed = False

    def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        self.calls.append((url, params))
        return _FakeResponse(self._payload)

    def close(self) -> None:
        self.is_closed = True


class _FakeBatchCall:
    def __init__(self, balances: list[int]) -> None:
        self._balances = balances

    def call(self) -> list[int]:
        return list(self._balances)


class _FakeFunctions:
    def __init__(self, balances: list[int]) -> None:
        self._balances = balances
        self.calls: list[tuple[list[str], list[int]]] = []

    def balanceOfBatch(self, accounts: list[str], ids: list[int]) -> _FakeBatchCall:
        self.calls.append((list(accounts), list(ids)))
        return _FakeBatchCall(self._balances)


class _FakeContract:
    def __init__(self, balances: list[int]) -> None:
        self.functions = _FakeFunctions(balances)


class _FakeWeb3:
    @staticmethod
    def to_checksum_address(value: str) -> str:
        return value.lower()


def _build_market_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "ignore-me",
            "conditionId": "0x" + "1" * 64,
            "question": "Decoy market",
            "marketMakerAddress": "0xdecoy000000000000000000000000000000000000",
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["11", "22"]',
            "fpmmLive": True,
        },
        {
            "id": "target-market",
            "conditionId": "0x" + "a" * 64,
            "question": "Will reserve fetch work?",
            "marketMakerAddress": "0x1234567890abcdef1234567890abcdef12345678",
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["101", "202"]',
            "fpmmLive": True,
        },
    ]


def test_get_pool_reserves_resolves_metadata_and_decodes_balances() -> None:
    http_client = _FakeHttpClient(_build_market_payload())
    contract = _FakeContract([12_500_000, 7_250_000])
    client = AlchemyRpcClient(
        alchemy_rpc_url="https://alchemy.example",
        http_client=http_client,
        ctf_contract=contract,
        web3=_FakeWeb3(),
        min_gamma_interval_s=0.0,
        min_rpc_interval_s=0.0,
    )

    reserves = client.get_pool_reserves("0x" + "a" * 64)

    assert reserves.condition_id == "0x" + "a" * 64
    assert reserves.market_maker_address == "0x1234567890abcdef1234567890abcdef12345678"
    assert reserves.yes_token_id == "101"
    assert reserves.no_token_id == "202"
    assert reserves.yes_reserve_raw == 12_500_000
    assert reserves.no_reserve_raw == 7_250_000
    assert str(reserves.yes_reserve) == "12.5"
    assert str(reserves.no_reserve) == "7.25"
    assert http_client.calls[0][1] == {"condition_id": "0x" + "a" * 64}
    assert contract.functions.calls == [
        (
            [
                "0x1234567890abcdef1234567890abcdef12345678",
                "0x1234567890abcdef1234567890abcdef12345678",
            ],
            [101, 202],
        )
    ]


def test_get_pool_reserves_reuses_metadata_but_requeries_reserves() -> None:
    http_client = _FakeHttpClient(_build_market_payload())
    contract = _FakeContract([1_000_000, 2_000_000])
    client = AlchemyRpcClient(
        alchemy_rpc_url="https://alchemy.example",
        http_client=http_client,
        ctf_contract=contract,
        web3=_FakeWeb3(),
        min_gamma_interval_s=0.0,
        min_rpc_interval_s=0.0,
        metadata_ttl_s=60.0,
        reserve_ttl_s=60.0,
    )

    first = client.get_pool_reserves("0x" + "a" * 64)
    second = client.get_pool_reserves("0x" + "a" * 64)

    assert first.condition_id == second.condition_id
    assert first.market_maker_address == second.market_maker_address
    assert first.yes_token_id == second.yes_token_id
    assert first.no_token_id == second.no_token_id
    assert first.yes_reserve_raw == second.yes_reserve_raw
    assert first.no_reserve_raw == second.no_reserve_raw
    assert len(http_client.calls) == 1
    assert len(contract.functions.calls) == 2


def test_get_pool_reserves_rejects_non_binary_yes_no_market() -> None:
    http_client = _FakeHttpClient(
        [
            {
                "id": "scalar-market",
                "conditionId": "0x" + "b" * 64,
                "question": "How high?",
                "marketMakerAddress": "0x1234567890abcdef1234567890abcdef12345678",
                "outcomes": '["Long", "Short"]',
                "clobTokenIds": '["111", "222"]',
                "fpmmLive": True,
            }
        ]
    )
    contract = _FakeContract([0, 0])
    client = AlchemyRpcClient(
        alchemy_rpc_url="https://alchemy.example",
        http_client=http_client,
        ctf_contract=contract,
        web3=_FakeWeb3(),
        min_gamma_interval_s=0.0,
        min_rpc_interval_s=0.0,
    )

    with pytest.raises(ValueError, match="binary YES/NO market"):
        client.get_pool_reserves("0x" + "b" * 64)


def test_live_path_requires_alchemy_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALCHEMY_POLYGON_RPC_URL", raising=False)

    with pytest.raises(EnvironmentError, match="ALCHEMY_POLYGON_RPC_URL"):
        AlchemyRpcClient()