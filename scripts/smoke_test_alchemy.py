from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.alchemy_rpc_client import AlchemyRpcClient

DEFAULT_CONDITION_ID = "0xe3b423dfad8c22ff75c9899c4e8176f628cf4ad4caa00481764d320e7415f7a9"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test the Alchemy/Gamma Polymarket reserve fetch path."
    )
    parser.add_argument(
        "--condition-id",
        default=DEFAULT_CONDITION_ID,
        help="Polymarket binary condition_id to probe.",
    )
    parser.add_argument(
        "--rpc-url",
        default="",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    rpc_url = os.getenv("ALCHEMY_POLYGON_RPC_URL", "").strip()
    if not rpc_url:
        raise EnvironmentError("ALCHEMY_POLYGON_RPC_URL must be set for the smoke test")

    with AlchemyRpcClient() as client:
        started_at_1 = time.perf_counter()
        reserves_1 = client.get_pool_reserves(args.condition_id)
        latency_ms_1 = (time.perf_counter() - started_at_1) * 1000.0

        started_at_2 = time.perf_counter()
        reserves_2 = client.get_pool_reserves(args.condition_id)
        latency_ms_2 = (time.perf_counter() - started_at_2) * 1000.0

    print(f"rpc_url={rpc_url}")
    print(f"condition_id={reserves_1.condition_id}")
    print(f"market_maker_address={reserves_1.market_maker_address}")
    print(f"yes_token_id={reserves_1.yes_token_id}")
    print(f"no_token_id={reserves_1.no_token_id}")
    print(f"yes_reserve_raw={reserves_1.yes_reserve_raw}")
    print(f"no_reserve_raw={reserves_1.no_reserve_raw}")
    print(f"yes_reserve={reserves_1.yes_reserve}")
    print(f"no_reserve={reserves_1.no_reserve}")
    print(f"latency_ms_1={latency_ms_1:.2f}")
    print(f"latency_ms_2={latency_ms_2:.2f}")
    if (
        reserves_1.condition_id != reserves_2.condition_id
        or reserves_1.market_maker_address != reserves_2.market_maker_address
        or reserves_1.yes_token_id != reserves_2.yes_token_id
        or reserves_1.no_token_id != reserves_2.no_token_id
        or reserves_1.yes_reserve_raw != reserves_2.yes_reserve_raw
        or reserves_1.no_reserve_raw != reserves_2.no_reserve_raw
    ):
        raise RuntimeError("warm smoke test returned inconsistent reserves across back-to-back calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())