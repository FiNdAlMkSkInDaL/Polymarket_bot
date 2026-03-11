import inspect
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams

# Show BalanceAllowanceParams fields
print("=== BalanceAllowanceParams ===")
print(inspect.getsource(BalanceAllowanceParams))

# Show update_balance_allowance source
print("\n=== update_balance_allowance ===")
print(inspect.getsource(ClobClient.update_balance_allowance))

# Show get_balance_allowance source
print("\n=== get_balance_allowance ===")
print(inspect.getsource(ClobClient.get_balance_allowance))
