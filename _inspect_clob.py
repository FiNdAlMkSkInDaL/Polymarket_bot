from py_clob_client.client import ClobClient
import inspect

for name in sorted(dir(ClobClient)):
    if any(k in name.lower() for k in ("approv", "allow", "balance", "set_")):
        method = getattr(ClobClient, name)
        sig = ""
        try:
            sig = str(inspect.signature(method))
        except Exception:
            pass
        print(f"  {name}{sig}")
