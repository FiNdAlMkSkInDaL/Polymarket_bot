import sys
sys.path.insert(0, "/home/botuser/polymarket-bot")
from src.core.config import settings
from web3 import Web3

w3 = Web3(Web3.HTTPProvider(settings.polygon_rpc_url))
from eth_account import Account
wallet = Account.from_key(settings.eoa_private_key).address

matic = w3.eth.get_balance(wallet)
print(f"Wallet: {wallet}")
print(f"MATIC (native): {Web3.from_wei(matic, 'ether')} MATIC")

USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
abi = [{"constant":True,"inputs":[{"name":"","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},
       {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"}]
usdc = w3.eth.contract(address=USDC_E, abi=abi)
bal = usdc.functions.balanceOf(wallet).call()
print(f"USDC.e balance : {bal / 10**6} USDC.e")
