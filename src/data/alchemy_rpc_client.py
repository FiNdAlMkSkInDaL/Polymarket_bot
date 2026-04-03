"""Read-only Polymarket AMM reserve client backed by Gamma + Alchemy Polygon RPC.

The resolver path is:

1. Query Gamma for a market by ``condition_id``.
2. Extract the FPMM market-maker address and YES/NO token ids.
3. Read both ERC-1155 balances from the CTF contract via a single
   ``balanceOfBatch`` call over ``eth_call``.

This module deliberately does not sign transactions or submit state-changing
requests. It is only for lightweight reserve inspection.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx
import requests
from requests.adapters import HTTPAdapter

try:
    from web3 import HTTPProvider, Web3
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without web3 installed
    HTTPProvider = None
    Web3 = None

from src.core.logger import get_logger

log = get_logger(__name__)

DEFAULT_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_CTF_CONTRACT_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
DEFAULT_TOKEN_DECIMALS = 6
DEFAULT_METADATA_TTL_S = 300.0
DEFAULT_RESERVE_TTL_S = 5.0
DEFAULT_MIN_RPC_INTERVAL_S = 0.25
DEFAULT_MIN_GAMMA_INTERVAL_S = 0.25
DEFAULT_TIMEOUT_S = 5.0

_ERC1155_CTF_ABI = [
    {
        "inputs": [
            {"internalType": "address[]", "name": "accounts", "type": "address[]"},
            {"internalType": "uint256[]", "name": "ids", "type": "uint256[]"},
        ],
        "name": "balanceOfBatch",
        "outputs": [{"internalType": "uint256[]", "name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function",
    }
]


def _normalize_condition_id(value: str) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        raise ValueError("condition_id is required")
    if not normalized.startswith("0x"):
        normalized = "0x" + normalized
    body = normalized[2:]
    if len(body) != 64:
        raise ValueError(f"condition_id must be 32 bytes / 64 hex chars; got {value!r}")
    int(body, 16)
    return normalized


def _parse_gamma_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _scale_uint(raw_value: int, decimals: int) -> Decimal:
    return Decimal(raw_value) / (Decimal(10) ** decimals)


@dataclass(frozen=True)
class MarketPoolMetadata:
    """Metadata required to read a market maker's outcome reserves."""

    condition_id: str
    market_id: str
    question: str
    market_maker_address: str
    yes_token_id: str
    no_token_id: str
    fpmm_live: bool
    fetched_at: float


@dataclass(frozen=True)
class PoolReserves:
    """Decoded YES/NO reserves for a binary Polymarket market."""

    condition_id: str
    market_maker_address: str
    yes_token_id: str
    no_token_id: str
    yes_reserve_raw: int
    no_reserve_raw: int
    token_decimals: int
    yes_reserve: Decimal
    no_reserve: Decimal
    fetched_at: float


class AlchemyRpcClient:
    """Lightweight read-only client for Polymarket FPMM reserve inspection.

    ``get_pool_reserves(condition_id)`` only supports binary YES/NO markets.
    Markets without a Gamma ``marketMakerAddress`` are treated as unsupported.
    """

    def __init__(
        self,
        *,
        alchemy_rpc_url: str | None = None,
        gamma_base_url: str = DEFAULT_GAMMA_BASE_URL,
        metadata_ttl_s: float = DEFAULT_METADATA_TTL_S,
        reserve_ttl_s: float = DEFAULT_RESERVE_TTL_S,
        min_rpc_interval_s: float = DEFAULT_MIN_RPC_INTERVAL_S,
        min_gamma_interval_s: float = DEFAULT_MIN_GAMMA_INTERVAL_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        token_decimals: int = DEFAULT_TOKEN_DECIMALS,
        ctf_contract_address: str = DEFAULT_CTF_CONTRACT_ADDRESS,
        http_client: httpx.Client | None = None,
        web3: Any | None = None,
        ctf_contract: Any | None = None,
    ) -> None:
        rpc_url = os.getenv("ALCHEMY_POLYGON_RPC_URL", "").strip()
        if not rpc_url and ctf_contract is None:
            raise EnvironmentError(
                "ALCHEMY_POLYGON_RPC_URL is required for live Polygon RPC access"
            )

        self._gamma_base_url = gamma_base_url.rstrip("/")
        self._metadata_ttl_s = max(0.0, float(metadata_ttl_s))
        self._reserve_ttl_s = max(0.0, float(reserve_ttl_s))
        self._min_rpc_interval_s = max(0.0, float(min_rpc_interval_s))
        self._min_gamma_interval_s = max(0.0, float(min_gamma_interval_s))
        self._token_decimals = max(0, int(token_decimals))
        self._rpc_url = rpc_url

        self._http_client = http_client or httpx.Client(timeout=timeout_s)
        self._owns_http_client = http_client is None
        self._rpc_session: requests.Session | None = None

        self._web3 = web3
        if ctf_contract is not None:
            self._ctf_contract = ctf_contract
        else:
            if Web3 is None or HTTPProvider is None:
                raise ModuleNotFoundError(
                    "web3 is required to use the live Alchemy RPC path"
                )
            self._rpc_session = requests.Session()
            adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8)
            self._rpc_session.mount("https://", adapter)
            self._rpc_session.mount("http://", adapter)
            self._web3 = self._web3 or Web3(
                HTTPProvider(
                    rpc_url,
                    request_kwargs={"timeout": timeout_s},
                    session=self._rpc_session,
                )
            )
            checksum_ctf = self._to_checksum_address(ctf_contract_address)
            self._ctf_contract = self._web3.eth.contract(
                address=checksum_ctf,
                abi=_ERC1155_CTF_ABI,
            )

        self._metadata_cache: dict[str, tuple[MarketPoolMetadata, float]] = {}

        self._gamma_rate_lock = threading.Lock()
        self._rpc_rate_lock = threading.Lock()
        self._next_gamma_at = 0.0
        self._next_rpc_at = 0.0

    def close(self) -> None:
        if self._owns_http_client and not self._http_client.is_closed:
            self._http_client.close()
        if self._rpc_session is not None:
            self._rpc_session.close()
            self._rpc_session = None

    def __enter__(self) -> "AlchemyRpcClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def get_market_metadata(
        self,
        condition_id: str,
        *,
        force_refresh: bool = False,
    ) -> MarketPoolMetadata:
        normalized_condition_id = _normalize_condition_id(condition_id)
        now = time.time()
        cached = self._metadata_cache.get(normalized_condition_id)
        if (
            not force_refresh
            and cached is not None
            and (now - cached[1]) < self._metadata_ttl_s
        ):
            return cached[0]

        payload = self._fetch_gamma_market(normalized_condition_id)
        metadata = self._extract_market_metadata(payload, normalized_condition_id)
        self._metadata_cache[normalized_condition_id] = (metadata, now)
        return metadata

    def get_pool_reserves(
        self,
        condition_id: str,
        *,
        force_refresh: bool = False,
    ) -> PoolReserves:
        normalized_condition_id = _normalize_condition_id(condition_id)

        metadata = self.get_market_metadata(normalized_condition_id, force_refresh=force_refresh)
        yes_raw, no_raw = self._fetch_pool_balances(metadata)
        fetched_at = time.time()
        return PoolReserves(
            condition_id=metadata.condition_id,
            market_maker_address=metadata.market_maker_address,
            yes_token_id=metadata.yes_token_id,
            no_token_id=metadata.no_token_id,
            yes_reserve_raw=yes_raw,
            no_reserve_raw=no_raw,
            token_decimals=self._token_decimals,
            yes_reserve=_scale_uint(yes_raw, self._token_decimals),
            no_reserve=_scale_uint(no_raw, self._token_decimals),
            fetched_at=fetched_at,
        )

    def _fetch_gamma_market(self, condition_id: str) -> list[dict[str, Any]]:
        self._respect_rate_limit(self._gamma_rate_lock, "_next_gamma_at", self._min_gamma_interval_s)
        response = self._http_client.get(
            f"{self._gamma_base_url}/markets",
            params={"condition_ids": condition_id},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise TypeError(f"unexpected Gamma response type: {type(payload)!r}")
        return [item for item in payload if isinstance(item, dict)]

    def _extract_market_metadata(
        self,
        payload: list[dict[str, Any]],
        condition_id: str,
    ) -> MarketPoolMetadata:
        matches = [
            item
            for item in payload
            if _normalize_condition_id(str(item.get("conditionId", ""))) == condition_id
        ]
        if not matches:
            raise LookupError(f"no Gamma market found for condition_id={condition_id}")

        market = next(
            (item for item in matches if item.get("marketMakerAddress")),
            matches[0],
        )
        market_maker_address = str(market.get("marketMakerAddress") or "").strip()
        if not market_maker_address:
            raise LookupError(
                f"market {market.get('id', '<unknown>')} has no marketMakerAddress"
            )

        yes_token_id, no_token_id = self._extract_binary_token_ids(market)
        fetched_at = time.time()
        return MarketPoolMetadata(
            condition_id=condition_id,
            market_id=str(market.get("id") or ""),
            question=str(market.get("question") or ""),
            market_maker_address=self._to_checksum_address(market_maker_address),
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            fpmm_live=bool(market.get("fpmmLive", False)),
            fetched_at=fetched_at,
        )

    def _extract_binary_token_ids(self, market: dict[str, Any]) -> tuple[str, str]:
        native_tokens = market.get("tokens")
        tokens: list[dict[str, str]] = []
        if isinstance(native_tokens, list) and native_tokens:
            tokens = [
                {
                    "token_id": str(token.get("token_id") or token.get("id") or ""),
                    "outcome": str(token.get("outcome") or ""),
                }
                for token in native_tokens
                if isinstance(token, dict)
            ]
        else:
            clob_ids = _parse_gamma_list(market.get("clobTokenIds"))
            outcomes = _parse_gamma_list(market.get("outcomes"))
            tokens = [
                {"token_id": str(token_id), "outcome": str(outcome)}
                for token_id, outcome in zip(clob_ids, outcomes)
            ]

        if len(tokens) != 2:
            raise ValueError(
                f"expected exactly 2 outcome tokens for binary market; found {len(tokens)}"
            )

        yes_token = None
        no_token = None
        for token in tokens:
            outcome = token.get("outcome", "").strip().upper()
            if outcome == "YES":
                yes_token = token
            elif outcome == "NO":
                no_token = token

        if yes_token is None or no_token is None:
            raise ValueError(
                f"condition_id={market.get('conditionId', '')} is not a binary YES/NO market"
            )

        yes_token_id = yes_token.get("token_id", "")
        no_token_id = no_token.get("token_id", "")
        if not yes_token_id or not no_token_id:
            raise ValueError("Gamma market payload is missing YES/NO token ids")
        return yes_token_id, no_token_id

    def _fetch_pool_balances(self, metadata: MarketPoolMetadata) -> tuple[int, int]:
        self._respect_rate_limit(self._rpc_rate_lock, "_next_rpc_at", self._min_rpc_interval_s)
        accounts = [metadata.market_maker_address, metadata.market_maker_address]
        ids = [int(metadata.yes_token_id), int(metadata.no_token_id)]
        raw_balances = self._ctf_contract.functions.balanceOfBatch(accounts, ids).call()
        if not isinstance(raw_balances, (list, tuple)) or len(raw_balances) != 2:
            raise TypeError(f"unexpected balanceOfBatch result: {raw_balances!r}")
        return int(raw_balances[0]), int(raw_balances[1])

    def _respect_rate_limit(self, lock: threading.Lock, attr_name: str, min_interval_s: float) -> None:
        if min_interval_s <= 0.0:
            return
        with lock:
            now = time.monotonic()
            next_allowed = getattr(self, attr_name)
            if next_allowed > now:
                time.sleep(next_allowed - now)
                now = time.monotonic()
            setattr(self, attr_name, now + min_interval_s)

    def _to_checksum_address(self, value: str) -> str:
        if self._web3 is not None and hasattr(self._web3, "to_checksum_address"):
            return self._web3.to_checksum_address(value)
        if Web3 is None:
            raise ModuleNotFoundError("web3 is required to checksum live addresses")
        return Web3.to_checksum_address(value)


def get_pool_reserves(
    condition_id: str,
    *,
    alchemy_rpc_url: str | None = None,
    gamma_base_url: str = DEFAULT_GAMMA_BASE_URL,
    metadata_ttl_s: float = DEFAULT_METADATA_TTL_S,
    reserve_ttl_s: float = DEFAULT_RESERVE_TTL_S,
    min_rpc_interval_s: float = DEFAULT_MIN_RPC_INTERVAL_S,
    min_gamma_interval_s: float = DEFAULT_MIN_GAMMA_INTERVAL_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    token_decimals: int = DEFAULT_TOKEN_DECIMALS,
    ctf_contract_address: str = DEFAULT_CTF_CONTRACT_ADDRESS,
) -> PoolReserves:
    """Convenience wrapper for one-shot reserve fetches."""

    with AlchemyRpcClient(
        alchemy_rpc_url=alchemy_rpc_url,
        gamma_base_url=gamma_base_url,
        metadata_ttl_s=metadata_ttl_s,
        reserve_ttl_s=reserve_ttl_s,
        min_rpc_interval_s=min_rpc_interval_s,
        min_gamma_interval_s=min_gamma_interval_s,
        timeout_s=timeout_s,
        token_decimals=token_decimals,
        ctf_contract_address=ctf_contract_address,
    ) as client:
        return client.get_pool_reserves(condition_id)
