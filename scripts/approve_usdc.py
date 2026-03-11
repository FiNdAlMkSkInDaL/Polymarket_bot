#!/usr/bin/env python3
"""
Approve USDC.e spending for Polymarket CTF Exchange contracts on Polygon.

Grants max-uint256 approval to both the standard CTF Exchange and the
Neg-Risk CTF Exchange so the bot can place real trades.

Usage:
    source .venv/bin/activate
    python scripts/approve_usdc.py
"""

from __future__ import annotations

import sys
import time

from web3 import Web3
from eth_account import Account

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, "/home/botuser/polymarket-bot")
from src.core.config import settings  # loads /dev/shm/secrets/.env

# ── constants ────────────────────────────────────────────────────────
CHAIN_ID = 137  # Polygon mainnet

USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")

SPENDERS = {
    "CTF Exchange": Web3.to_checksum_address(
        "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    ),
    "Neg-Risk CTF Exchange": Web3.to_checksum_address(
        "0xC5d563A36AE78145C45a50134d48A1215220f80a"
    ),
}

MAX_UINT256 = 2**256 - 1

# Minimal ERC-20 ABI (only approve + allowance)
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
]


def main() -> None:
    # ── connect to Polygon ───────────────────────────────────────────
    rpc_url = settings.polygon_rpc_url
    if not rpc_url:
        sys.exit("ERROR: POLYGON_RPC_URL is not set in secrets.")

    pk = settings.eoa_private_key
    if not pk:
        sys.exit("ERROR: EOA_PRIVATE_KEY is not set in secrets.")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        sys.exit(f"ERROR: Cannot connect to RPC at {rpc_url}")

    account = Account.from_key(pk)
    wallet = account.address
    print(f"Wallet   : {wallet}")
    print(f"RPC      : {rpc_url}")
    print(f"Chain ID : {CHAIN_ID}")
    print()

    usdc = w3.eth.contract(address=USDC_E, abi=ERC20_ABI)

    for label, spender in SPENDERS.items():
        # ── check current allowance ──────────────────────────────────
        current = usdc.functions.allowance(wallet, spender).call()
        print(f"[{label}]")
        print(f"  Spender          : {spender}")
        print(f"  Current allowance: {current}")

        if current >= MAX_UINT256 // 2:
            print("  ✓ Already approved (sufficient allowance). Skipping.\n")
            continue

        # ── pre-flight simulation ────────────────────────────────────
        try:
            usdc.functions.approve(spender, MAX_UINT256).call({"from": wallet})
            print("  Pre-flight       : OK")
        except Exception as e:
            print(f"  Pre-flight FAILED: {e}")
            print("  Skipping this spender.\n")
            continue

        # ── build approve tx (legacy type for Polygon reliability) ───
        nonce = w3.eth.get_transaction_count(wallet, "pending")
        gas_price = max(w3.eth.gas_price, w3.to_wei(50, "gwei"))
        tx = usdc.functions.approve(spender, MAX_UINT256).build_transaction(
            {
                "chainId": CHAIN_ID,
                "from": wallet,
                "nonce": nonce,
                "gas": 100_000,
                "gasPrice": gas_price,
            }
        )

        signed = w3.eth.account.sign_transaction(tx, private_key=pk)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  Tx hash          : {tx_hash.hex()}")
        print("  Waiting for receipt …", end="", flush=True)

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        status = receipt["status"]
        gas_used = receipt["gasUsed"]
        print(f" block {receipt['blockNumber']}, gas={gas_used}, "
              f"status={'SUCCESS' if status == 1 else 'FAILED'}")

        if status != 1:
            print(f"  ✗ Transaction REVERTED. Check on Polygonscan.")
            sys.exit(1)
        else:
            new_allowance = usdc.functions.allowance(wallet, spender).call()
            print(f"  New allowance    : {new_allowance}")
            print(f"  ✓ Approved successfully.")
        print()

    # ── notify CLOB backend to refresh cached allowances ─────────────
    print("Notifying CLOB backend to refresh allowances …")
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType

        creds = ApiCreds(
            api_key=settings.polymarket_api_key,
            api_secret=settings.polymarket_secret,
            api_passphrase=settings.polymarket_passphrase,
        )
        clob = ClobClient(
            settings.clob_http_url,
            key=pk,
            chain_id=CHAIN_ID,
            creds=creds,
        )

        for asset_type in [AssetType.COLLATERAL, AssetType.CONDITIONAL]:
            params = BalanceAllowanceParams(asset_type=asset_type)
            resp = clob.update_balance_allowance(params)
            print(f"  update_balance_allowance({asset_type}): {resp}")
    except Exception as exc:
        print(f"  Warning: CLOB refresh failed ({exc}). The exchange will pick up "
              "the on-chain allowance on next order attempt.")

    print("\nDone.")


if __name__ == "__main__":
    main()
