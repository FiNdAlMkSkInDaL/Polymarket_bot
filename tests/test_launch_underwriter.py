from __future__ import annotations

import json

from scripts import launch_underwriter


def test_load_active_targets_preserves_pre_resolved_token_ids(tmp_path) -> None:
    payload = {
        "active_markets": [
            {
                "condition_id": "0x" + "1" * 64,
                "question": "Will BTC hit 200k?",
                "category": "crypto",
                "entry_yes_ask": 0.041,
                "yes_token_id": "yes-token-1",
                "no_token_id": "no-token-1",
            }
        ]
    }
    path = tmp_path / "live_targets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    targets = launch_underwriter.load_active_targets(path)

    assert len(targets) == 1
    assert targets[0].yes_token_id == "yes-token-1"
    assert targets[0].no_token_id == "no-token-1"


def test_resolve_targets_via_gamma_skips_http_when_tokens_already_present(monkeypatch) -> None:
    class _Session:
        def get(self, *args, **kwargs):
            raise AssertionError("Gamma should not be called when token ids are already present")

    monkeypatch.setattr(launch_underwriter.requests, "Session", lambda: _Session())
    target = launch_underwriter.ActiveTarget(
        condition_id="0x" + "2" * 64,
        question="Will ETH hit 20k?",
        category="crypto",
        entry_yes_ask=launch_underwriter.Decimal("0.03"),
        terminal_yes_ask=None,
        max_yes_ask=None,
        max_no_drawdown_cents=None,
        yes_token_id="yes-token-2",
        no_token_id="no-token-2",
    )

    resolved = launch_underwriter.resolve_targets_via_gamma([target], timeout_seconds=1.0)

    assert resolved[target.condition_id].yes_token_id == "yes-token-2"
    assert resolved[target.condition_id].no_token_id == "no-token-2"