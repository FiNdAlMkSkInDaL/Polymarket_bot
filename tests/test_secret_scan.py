from __future__ import annotations

from pathlib import Path

from src.tools.secret_scan import ROOT, Finding, has_forbidden_secret_filename, scan_text


def _scan(rel_path: str, text: str) -> list[Finding]:
    return scan_text(text, ROOT / rel_path)


def test_flags_hardcoded_env_secret_assignment() -> None:
    findings = _scan(
        "scripts/example.sh",
        'ORACLE_SPORTS_API_KEY="869484bcf56587d333eabda03e41dd2d"\n',  # secret-scan: ignore
    )

    assert [finding.message for finding in findings] == ["hardcoded value for ORACLE_SPORTS_API_KEY"]


def test_flags_hardcoded_secret_in_sed_replacement() -> None:
    findings = _scan(
        "scripts/example.sh",
        "sed -i -E 's|^ORACLE_SPORTS_API_KEY=.*|ORACLE_SPORTS_API_KEY=869484bcf56587d333eabda03e41dd2d|' .env\n",  # secret-scan: ignore
    )

    assert [finding.message for finding in findings] == ["hardcoded value for ORACLE_SPORTS_API_KEY"]


def test_ignores_redacted_value() -> None:
    findings = _scan(
        "scripts/example.sh",
        "sed -i -E 's|^ORACLE_SPORTS_API_KEY=.*|ORACLE_SPORTS_API_KEY=***REMOVED***|' .env\n",
    )

    assert findings == []


def test_ignores_placeholder_diagnostics() -> None:
    findings = _scan(
        "scripts/run_diagnostics.py",
        'telegram_bot_token="1234567890:AAHdqTcvZE_FAKE_TOKEN"\n',
    )

    assert findings == []


def test_ignores_env_name_lookup() -> None:
    findings = _scan(
        "src/core/config.py",
        'oracle_sports_api_key: str = _env("ORACLE_SPORTS_API_KEY")\n',
    )

    assert findings == []


def test_ignores_public_wallet_addresses() -> None:
    findings = _scan(
        "src/signals/whale_monitor.py",
        'DEFAULT_WHALE_WALLETS = ["0x1076e14139e0e0B2F1f23E379Bb7b45E30Cb4e26"]\n',
    )

    assert findings == []


def test_flags_telegram_token_literal() -> None:
    findings = _scan(
        "scripts/example.py",
        'telegram_bot_token = "8723760057:AAGWZIVXIA0T5e3221YQi7nd4hHkfxEJjcw"\n',  # secret-scan: ignore
    )

    assert [finding.message for finding in findings] == ["hardcoded value for telegram_bot_token"]


def test_ignores_known_test_private_key_fixture() -> None:
    findings = _scan(
        "tests/test_polymarket_clob_adapter.py",
        'private_key="0x59c6995e998f97a5a0044966f0945382dbf59596e17f7b7b6b6d3d4d6f8d2f4c"\n',
    )

    assert findings == []


def test_forbidden_secret_filename_blocks_tracked_env_file() -> None:
    assert has_forbidden_secret_filename(ROOT / ".env") is True
    assert has_forbidden_secret_filename(ROOT / ".env.example") is False
    assert has_forbidden_secret_filename(ROOT / "config/app.env.template") is False