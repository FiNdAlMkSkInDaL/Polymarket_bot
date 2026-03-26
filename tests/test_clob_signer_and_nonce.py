from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any

import pytest
from py_clob_client.config import get_contract_config

from src.execution.clob_signer import ClobSigner
from src.execution.nonce_manager import ClobNonceManager
from src.execution.polymarket_clob_translator import ClobOrderIntent, ClobTimeInForce


_PRIVATE_KEY = "0x59c6995e998f97a5a0044966f0945382dbf59596e17f7b7b6b6d3d4d6f8d2f4c"
_EXCHANGE = get_contract_config(137).exchange
_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def _payload(**overrides: Any) -> dict[str, str | bool]:
    payload: dict[str, str | bool] = {
        "conditionId": "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "token_id": "123456789",
        "client_order_id": "CID-1",
        "price": "0.640001",
        "size": "42.500000",
        "side": "BUY",
        "fee_rate_bps": "0",
        "nonce": "7",
        "expiration": "1700000000",
        "taker": _ZERO_ADDRESS,
        "post_only": False,
    }
    payload.update(overrides)
    return payload


def _intent(**overrides: Any) -> ClobOrderIntent:
    values = {
        "condition_id": "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "token_id": "123456789",
        "outcome": "YES",
        "action": "BUY",
        "price": Decimal("0.640001"),
        "size": Decimal("42.500000"),
        "time_in_force": ClobTimeInForce.GTC,
        "client_order_id": "CID-1",
        "post_only": False,
        "fee_rate_bps": 0,
        "nonce": 7,
        "expiration": 1700000000,
    }
    values.update(overrides)
    return ClobOrderIntent(**values)


def _signer() -> ClobSigner:
    return ClobSigner(
        private_key=_PRIVATE_KEY,
        chain_id=137,
        exchange_address=_EXCHANGE,
    )


def _assert_no_floats(value: Any) -> None:
    if isinstance(value, float):
        raise AssertionError("float leaked into signed order")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_floats(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_floats(nested)


def test_nonce_manager_rejects_negative_starting_nonce() -> None:
    with pytest.raises(ValueError, match="starting_nonce"):
        ClobNonceManager(-1)


def test_nonce_manager_reserves_single_nonce_by_default() -> None:
    manager = ClobNonceManager(5)

    assert manager.reserve_nonces() == (5,)
    assert manager.next_nonce == 6


def test_nonce_manager_reserves_contiguous_block() -> None:
    manager = ClobNonceManager(10)

    assert manager.reserve_nonces(3) == (10, 11, 12)
    assert manager.next_nonce == 13


def test_nonce_manager_reservations_continue_from_last_nonce() -> None:
    manager = ClobNonceManager(2)

    first = manager.reserve_nonces(2)
    second = manager.reserve_nonces(2)

    assert first == (2, 3)
    assert second == (4, 5)


def test_nonce_manager_sync_nonce_fast_forwards_next_nonce() -> None:
    manager = ClobNonceManager(1)

    manager.sync_nonce(8)

    assert manager.next_nonce == 9


def test_nonce_manager_sync_nonce_does_not_rewind() -> None:
    manager = ClobNonceManager(6)
    manager.reserve_nonces(2)

    manager.sync_nonce(1)

    assert manager.next_nonce == 8


def test_nonce_manager_rejects_non_positive_reservation_count() -> None:
    manager = ClobNonceManager()

    with pytest.raises(ValueError, match="count"):
        manager.reserve_nonces(0)


def test_nonce_manager_rejects_invalid_sync_nonce() -> None:
    manager = ClobNonceManager()

    with pytest.raises(ValueError, match="venue_nonce"):
        manager.sync_nonce(-1)


def test_nonce_manager_concurrent_reservations_remain_unique_and_contiguous() -> None:
    manager = ClobNonceManager(0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        reserved = list(executor.map(lambda _: manager.reserve_nonces(1)[0], range(20)))

    assert sorted(reserved) == list(range(20))
    assert manager.next_nonce == 20


def test_signer_uses_private_key_address_by_default() -> None:
    signer = _signer()

    assert signer.maker_address == signer.signer_address
    assert signer.signer_address == "0x6dfeF785Eb7c9C2Ac98A6d81dE2aA1791D2e8056"


def test_signer_rejects_invalid_chain_id() -> None:
    with pytest.raises(ValueError, match="chain_id"):
        ClobSigner(private_key=_PRIVATE_KEY, chain_id=0, exchange_address=_EXCHANGE)


def test_signer_rejects_non_string_decimal_inputs() -> None:
    signer = _signer()

    with pytest.raises(ValueError, match="price"):
        signer.sign_create_order_payload({**_payload(), "price": Decimal("0.64")})


def test_signer_buy_order_maps_maker_amount_to_quote_units() -> None:
    signer = _signer()

    signed_order = signer.sign_create_order_payload(_payload(side="BUY"))

    assert signed_order["makerAmount"] == "27200042"
    assert signed_order["takerAmount"] == "42500000"


def test_signer_sell_order_maps_maker_amount_to_size_units() -> None:
    signer = _signer()

    signed_order = signer.sign_create_order_payload(_payload(side="SELL"))

    assert signed_order["makerAmount"] == "42500000"
    assert signed_order["takerAmount"] == "27200042"
    assert signed_order["side"] == "SELL"


def test_signer_preserves_nonce_and_expiration_as_strings() -> None:
    signer = _signer()

    signed_order = signer.sign_create_order_payload(_payload(nonce="9", expiration="1700000010"))

    assert signed_order["nonce"] == "9"
    assert signed_order["expiration"] == "1700000010"


def test_signer_sign_intent_matches_create_order_payload_path() -> None:
    signer = _signer()

    from_intent = signer.sign_intent(_intent())
    from_payload = signer.sign_create_order_payload(_payload())

    assert from_intent == from_payload


def test_signer_is_deterministic_for_same_payload() -> None:
    signer = _signer()

    first = signer.sign_create_order_payload(_payload())
    second = signer.sign_create_order_payload(_payload())

    assert first == second


def test_signer_changes_signature_when_nonce_changes() -> None:
    signer = _signer()

    first = signer.sign_create_order_payload(_payload(nonce="7"))
    second = signer.sign_create_order_payload(_payload(nonce="8"))

    assert first["signature"] != second["signature"]
    assert first["salt"] != second["salt"]


def test_signer_changes_signature_when_client_order_id_changes() -> None:
    signer = _signer()

    first = signer.sign_create_order_payload(_payload(client_order_id="CID-1"))
    second = signer.sign_create_order_payload(_payload(client_order_id="CID-2"))

    assert first["signature"] != second["signature"]
    assert first["salt"] != second["salt"]


def test_signer_emits_expected_field_shape_without_floats() -> None:
    signer = _signer()

    signed_order = signer.sign_create_order_payload(_payload())

    assert set(signed_order) == {
        "salt",
        "maker",
        "signer",
        "taker",
        "tokenId",
        "makerAmount",
        "takerAmount",
        "expiration",
        "nonce",
        "feeRateBps",
        "side",
        "signatureType",
        "signature",
    }
    _assert_no_floats(signed_order)


def test_signer_uses_known_deterministic_signature_vector() -> None:
    signer = _signer()

    signed_order = signer.sign_create_order_payload(_payload())

    assert signed_order["salt"] == 56982879106618020023543382463103654388438853715309378636746846494864518083792
    assert signed_order["signature"] == "0xa72fdf4f8d7332ae77fd1011a25ca831b5f1c0df29c68326788cc978af573b0d1f39bd1fdaf1b0938e14c7ffccde5966f4ae27103f0ce886c5a9574bf2e5f46e1c"