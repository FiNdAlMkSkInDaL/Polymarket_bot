from __future__ import annotations

from scripts.validate_si10_relationships import load_relationships, validate_relationships


def test_validate_si10_relationships_accepts_active_markets(tmp_path):
    path = tmp_path / "si10.json"
    path.write_text(
        """
[
  {
    "relationship_id": "rel-1",
    "label": "Example",
    "base_a_condition_id": "A",
    "base_b_condition_id": "B",
    "joint_condition_id": "J"
  }
]
""".strip(),
        encoding="utf-8",
    )

    relationships = load_relationships(path)
    market_index = {
        "A": {
            "conditionId": "A",
            "question": "A?",
            "active": True,
            "closed": False,
            "acceptingOrders": True,
            "clobTokenIds": ["YES_A", "NO_A"],
            "outcomes": ["YES", "NO"],
            "volume24hr": 0,
            "liquidity": 0,
            "endDate": "2099-01-01T00:00:00Z"
        },
        "B": {
            "conditionId": "B",
            "question": "B?",
            "active": True,
            "closed": False,
            "acceptingOrders": True,
            "clobTokenIds": ["YES_B", "NO_B"],
            "outcomes": ["YES", "NO"],
            "volume24hr": 0,
            "liquidity": 0,
            "endDate": "2099-01-01T00:00:00Z"
        },
        "J": {
            "conditionId": "J",
            "question": "J?",
            "active": True,
            "closed": False,
            "acceptingOrders": True,
            "clobTokenIds": ["YES_J", "NO_J"],
            "outcomes": ["YES", "NO"],
            "volume24hr": 0,
            "liquidity": 0,
            "endDate": "2099-01-01T00:00:00Z"
        },
    }

    issues, resolved = validate_relationships(relationships, market_index)

    assert issues == []
    assert resolved["rel-1"]["markets"]["base_a_condition_id"]["yes_token_id"] == "YES_A"


def test_validate_si10_relationships_rejects_missing_and_inactive_markets(tmp_path):
    path = tmp_path / "si10.json"
    path.write_text(
        """
[
  {
    "relationship_id": "rel-1",
    "base_a_condition_id": "A",
    "base_b_condition_id": "A",
    "joint_condition_id": "J"
  }
]
""".strip(),
        encoding="utf-8",
    )

    relationships = load_relationships(path)
    market_index = {
        "A": {
            "conditionId": "A",
            "question": "A?",
            "active": False,
            "closed": True,
            "acceptingOrders": False,
            "clobTokenIds": ["YES_A", "NO_A"],
            "outcomes": ["YES", "NO"],
            "volume24hr": 0,
            "liquidity": 0,
            "endDate": "2099-01-01T00:00:00Z"
        }
    }

    issues, _resolved = validate_relationships(relationships, market_index)

    reasons = {(issue.field, issue.reason) for issue in issues}
    assert ("base_b_condition_id", "duplicate_condition_id") in reasons
    assert ("base_a_condition_id", "market_not_active") in reasons
    assert ("joint_condition_id", "condition_id_not_found") in reasons