from __future__ import annotations

import hashlib
from decimal import Decimal, ROUND_DOWN
from typing import Any, Literal, Mapping

from eth_account import Account
from eth_utils import keccak
from poly_eip712_structs import make_domain
from py_order_utils.model import Order
from py_order_utils.utils import prepend_zx

from src.execution.polymarket_clob_translator import ClobOrderIntent


_TOKEN_DECIMALS = Decimal("1000000")
_MICRO = Decimal("0.000001")
_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def _require_non_empty_string(name: str, value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isdigit() or (normalized.startswith("-") and normalized[1:].isdigit()):
            return int(normalized)
    raise ValueError(f"{name} must be an int")


def _require_decimal_string(name: str, value: Any) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a decimal string")
    decimal_value = Decimal(value.strip())
    if not decimal_value.is_finite() or decimal_value <= Decimal("0"):
        raise ValueError(f"{name} must be a strictly positive finite decimal string")
    return decimal_value


def _scale_decimal(value: Decimal) -> int:
    quantized = value.quantize(_MICRO, rounding=ROUND_DOWN)
    scaled = quantized * _TOKEN_DECIMALS
    if scaled != scaled.to_integral_value():
        raise ValueError("scaled decimal must be an integer number of token units")
    return int(scaled)


class ClobSigner:
    def __init__(
        self,
        *,
        private_key: str,
        chain_id: int,
        exchange_address: str,
        maker_address: str | None = None,
        signature_type: int = 0,
    ) -> None:
        normalized_private_key = _require_non_empty_string("private_key", private_key)
        if not isinstance(chain_id, int) or chain_id <= 0:
            raise ValueError("chain_id must be a positive int")
        if not isinstance(signature_type, int) or signature_type < 0:
            raise ValueError("signature_type must be a non-negative int")

        self._account = Account.from_key(normalized_private_key)
        self._private_key = normalized_private_key
        self._chain_id = chain_id
        self._exchange_address = _require_non_empty_string("exchange_address", exchange_address)
        self._maker_address = maker_address or self._account.address
        self._signature_type = signature_type
        self._domain = make_domain(
            name="Polymarket CTF Exchange",
            version="1",
            chainId=str(chain_id),
            verifyingContract=self._exchange_address,
        )

    @property
    def maker_address(self) -> str:
        return self._maker_address

    @property
    def signer_address(self) -> str:
        return self._account.address

    def sign_intent(self, intent: ClobOrderIntent) -> dict[str, Any]:
        price_units, size_units = self._amount_units(intent.action, intent.price, intent.size)
        signed_order = self._build_signed_order(
            token_id=intent.token_id,
            client_order_id=intent.client_order_id,
            side=intent.action,
            maker_amount=price_units,
            taker_amount=size_units,
            fee_rate_bps=intent.fee_rate_bps,
            nonce=intent.nonce,
            expiration=intent.expiration,
            taker=intent.taker,
        )
        return signed_order

    def sign_create_order_payload(self, payload: Mapping[str, str | bool]) -> dict[str, Any]:
        token_id = _require_non_empty_string("token_id", str(payload.get("token_id") or ""))
        client_order_id = _require_non_empty_string("client_order_id", str(payload.get("client_order_id") or ""))
        side = _require_non_empty_string("side", str(payload.get("side") or "")).upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported side: {side!r}")
        price = _require_decimal_string("price", payload.get("price"))
        size = _require_decimal_string("size", payload.get("size"))
        fee_rate_bps = _require_int("fee_rate_bps", payload.get("fee_rate_bps", "0"))
        nonce = _require_int("nonce", payload.get("nonce", "0"))
        expiration = _require_int("expiration", payload.get("expiration", "0"))
        taker = str(payload.get("taker") or _ZERO_ADDRESS)

        maker_amount, taker_amount = self._amount_units(side, price, size)
        return self._build_signed_order(
            token_id=token_id,
            client_order_id=client_order_id,
            side=side,
            maker_amount=maker_amount,
            taker_amount=taker_amount,
            fee_rate_bps=fee_rate_bps,
            nonce=nonce,
            expiration=expiration,
            taker=taker,
        )

    def _build_signed_order(
        self,
        *,
        token_id: str,
        client_order_id: str,
        side: Literal["BUY", "SELL"],
        maker_amount: int,
        taker_amount: int,
        fee_rate_bps: int,
        nonce: int,
        expiration: int,
        taker: str,
    ) -> dict[str, Any]:
        salt = self._deterministic_salt(
            token_id=token_id,
            client_order_id=client_order_id,
            side=side,
            maker_amount=maker_amount,
            taker_amount=taker_amount,
            fee_rate_bps=fee_rate_bps,
            nonce=nonce,
            expiration=expiration,
            taker=taker,
        )
        side_value = 0 if side == "BUY" else 1
        order = Order(
            salt=salt,
            maker=self._maker_address,
            signer=self.signer_address,
            taker=taker,
            tokenId=int(token_id),
            makerAmount=maker_amount,
            takerAmount=taker_amount,
            expiration=expiration,
            nonce=nonce,
            feeRateBps=fee_rate_bps,
            side=side_value,
            signatureType=self._signature_type,
        )
        struct_hash = prepend_zx(keccak(order.signable_bytes(domain=self._domain)).hex())
        signature = prepend_zx(Account._sign_hash(struct_hash, self._private_key).signature.hex())
        return {
            "salt": order["salt"],
            "maker": order["maker"],
            "signer": order["signer"],
            "taker": order["taker"],
            "tokenId": str(order["tokenId"]),
            "makerAmount": str(order["makerAmount"]),
            "takerAmount": str(order["takerAmount"]),
            "expiration": str(order["expiration"]),
            "nonce": str(order["nonce"]),
            "feeRateBps": str(order["feeRateBps"]),
            "side": side,
            "signatureType": order["signatureType"],
            "signature": signature,
        }

    def _amount_units(self, side: Literal["BUY", "SELL"], price: Decimal, size: Decimal) -> tuple[int, int]:
        size_units = _scale_decimal(size)
        proceeds_units = _scale_decimal(price * size)
        if side == "BUY":
            return proceeds_units, size_units
        return size_units, proceeds_units

    def _deterministic_salt(
        self,
        *,
        token_id: str,
        client_order_id: str,
        side: Literal["BUY", "SELL"],
        maker_amount: int,
        taker_amount: int,
        fee_rate_bps: int,
        nonce: int,
        expiration: int,
        taker: str,
    ) -> int:
        canonical = "\x1f".join(
            [
                str(self._chain_id),
                self._exchange_address.lower(),
                self._maker_address.lower(),
                self.signer_address.lower(),
                str(token_id),
                client_order_id,
                side,
                str(maker_amount),
                str(taker_amount),
                str(fee_rate_bps),
                str(nonce),
                str(expiration),
                taker.lower(),
                str(self._signature_type),
            ]
        )
        return int.from_bytes(hashlib.sha256(canonical.encode("ascii")).digest(), byteorder="big")