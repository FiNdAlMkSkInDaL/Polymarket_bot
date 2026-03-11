import importlib, pkgutil, os, sys

# Find py_clob_client package location
import py_clob_client
pkg_dir = os.path.dirname(py_clob_client.__file__)
print(f"Package dir: {pkg_dir}\n")

# List all files
for root, dirs, files in os.walk(pkg_dir):
    for f in files:
        if f.endswith('.py'):
            rel = os.path.relpath(os.path.join(root, f), pkg_dir)
            print(rel)

# Search for contract addresses / USDC references
print("\n=== Searching for contract addresses ===")
import re
for root, dirs, files in os.walk(pkg_dir):
    for f in files:
        if f.endswith('.py'):
            fpath = os.path.join(root, f)
            with open(fpath) as fh:
                content = fh.read()
            if re.search(r'0x[0-9a-fA-F]{40}', content):
                rel = os.path.relpath(fpath, pkg_dir)
                addrs = re.findall(r'0x[0-9a-fA-F]{40}', content)
                print(f"\n{rel}:")
                for a in set(addrs):
                    print(f"  {a}")

# Check if web3 is available
try:
    import web3
    print(f"\nweb3 version: {web3.__version__}")
except ImportError:
    print("\nweb3 NOT installed")
