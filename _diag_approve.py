"""Diagnose why the USDC.e approve() reverted."""
import sys
sys.path.insert(0, "/home/botuser/polymarket-bot")
from src.core.config import settings
from web3 import Web3
from eth_account import Account

w3 = Web3(Web3.HTTPProvider(settings.polygon_rpc_url))
wallet = Account.from_key(settings.eoa_private_key).address

USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
CTF_EXCHANGE = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
MAX_UINT256 = 2**256 - 1

ERC20_ABI = [
    {"constant":False,"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"},
    {"constant":True,"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"type":"function"},
    {"constant":True,"inputs":[{"name":"","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},
    {"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"type":"function"},
    {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},
    {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
]

usdc = w3.eth.contract(address=USDC_E, abi=ERC20_ABI)

print(f"Wallet: {wallet}")
print(f"MATIC balance: {Web3.from_wei(w3.eth.get_balance(wallet), 'ether')}")
print(f"Token name: {usdc.functions.name().call()}")
print(f"Token symbol: {usdc.functions.symbol().call()}")
print(f"Token decimals: {usdc.functions.decimals().call()}")
print(f"USDC.e balance: {usdc.functions.balanceOf(wallet).call() / 1e6}")
print(f"Current allowance to CTF: {usdc.functions.allowance(wallet, CTF_EXCHANGE).call()}")

# Try eth_call to simulate approve and get revert reason
print("\n--- Simulating approve via eth_call ---")
try:
    result = usdc.functions.approve(CTF_EXCHANGE, MAX_UINT256).call({"from": wallet})
    print(f"eth_call succeeded, returned: {result}")
except Exception as e:
    print(f"eth_call REVERTED: {e}")

# Try with smaller amount
print("\n--- Simulating approve with 10 USDC ---")
try:
    result = usdc.functions.approve(CTF_EXCHANGE, 10 * 10**6).call({"from": wallet})
    print(f"eth_call succeeded, returned: {result}")
except Exception as e:
    print(f"eth_call REVERTED: {e}")

# Check the failed tx receipts
for label, tx_hash in [
    ("CTF Exchange", "0x9d4badb221b7a223153898c7b143f6cd5e0987643c21e1ef8064f70d6c8affa5"),
    ("Neg-Risk", "0x9f3e0339a384e8c633e0b0ce88a034a63763de8be79f60a1ea96dc3e9b12d1f3"),
]:
    print(f"\n--- {label} tx receipt ---")
    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        print(f"  Status: {receipt['status']}")
        print(f"  Gas used: {receipt['gasUsed']}")
        print(f"  Block: {receipt['blockNumber']}")
        tx = w3.eth.get_transaction(tx_hash)
        print(f"  Gas limit: {tx['gas']}")
        print(f"  To: {tx['to']}")
        print(f"  Input data (first 10 chars): {tx['input'][:10]}...")

        # Try to replay the tx to get revert reason
        try:
            w3.eth.call(
                {"from": tx["from"], "to": tx["to"], "data": tx["input"], "value": tx.get("value", 0)},
                receipt["blockNumber"] - 1,
            )
            print("  Replay: no revert (odd!)")
        except Exception as e:
            print(f"  Replay revert reason: {e}")
    except Exception as e:
        print(f"  Error: {e}")
