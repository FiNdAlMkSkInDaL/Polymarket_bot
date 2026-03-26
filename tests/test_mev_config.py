from __future__ import annotations

import pytest

from src.core.config import Settings


def test_mev_config_defaults_load_without_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("POLYGON_RPC_URL", raising=False)
    monkeypatch.delenv("POLYGON_RPC_HTTP_URL", raising=False)
    monkeypatch.delenv("POLYGON_RPC_WSS_URL", raising=False)
    monkeypatch.delenv("UMA_OPTIMISTIC_ORACLE_ADDRESS", raising=False)
    monkeypatch.delenv("MEV_SHADOW_SWEEP_PREMIUM_PCT", raising=False)
    monkeypatch.delenv("MEV_D3_PANIC_THRESHOLD", raising=False)

    config = Settings()

    assert config.polygon_rpc_url == ""
    assert config.polygon_rpc_http_url == ""
    assert config.polygon_rpc_wss_url == ""
    assert config.uma_optimistic_oracle_address == ""
    assert config.mev_shadow_sweep_premium_pct == 0.03
    assert config.mev_d3_panic_threshold == 0.12


def test_mev_config_env_aliases_and_overrides(monkeypatch) -> None:
    monkeypatch.setenv("POLYGON_RPC_HTTP_URL", "https://polygon-rpc.example")
    monkeypatch.setenv("POLYGON_RPC_WSS_URL", "wss://polygon-rpc.example/ws")
    monkeypatch.setenv("UMA_OPTIMISTIC_ORACLE_ADDRESS", "0x123400000000000000000000000000000000abcd")
    monkeypatch.setenv("MEV_SHADOW_SWEEP_PREMIUM_PCT", "0.05")
    monkeypatch.setenv("MEV_D3_PANIC_THRESHOLD", "0.18")
    monkeypatch.delenv("POLYGON_RPC_URL", raising=False)

    config = Settings()

    assert config.polygon_rpc_http_url == "https://polygon-rpc.example"
    assert config.polygon_rpc_url == "https://polygon-rpc.example"
    assert config.polygon_rpc_wss_url == "wss://polygon-rpc.example/ws"
    assert config.uma_optimistic_oracle_address == "0x123400000000000000000000000000000000abcd"
    assert config.mev_shadow_sweep_premium_pct == 0.05
    assert config.mev_d3_panic_threshold == 0.18


@pytest.mark.parametrize(
    ("env_name", "env_value", "expected_message"),
    [
        ("MEV_D3_PANIC_THRESHOLD", "0", "MEV_D3_PANIC_THRESHOLD"),
        ("MEV_D3_PANIC_THRESHOLD", "1.0", "MEV_D3_PANIC_THRESHOLD"),
        ("MEV_SHADOW_SWEEP_PREMIUM_PCT", "-0.01", "MEV_SHADOW_SWEEP_PREMIUM_PCT"),
        ("MEV_SHADOW_SWEEP_PREMIUM_PCT", "1.0", "MEV_SHADOW_SWEEP_PREMIUM_PCT"),
    ],
)
def test_mev_config_invalid_percentages_hard_fail(monkeypatch, env_name: str, env_value: str, expected_message: str) -> None:
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValueError, match=expected_message):
        Settings()
