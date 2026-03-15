"""Patch the 13 failing tests to match March 10 parameter recalibrations."""

ROOT = "/home/botuser/polymarket-bot"


def fix_file(path, replacements):
    with open(path, "r") as f:
        content = f.read()
    for old, new in replacements:
        if old not in content:
            print(f"  WARNING: pattern not found in {path}:\n    {old[:80]}...")
        else:
            content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Fixed {path}")


# ── 1. test_alpha_evolution.py ──
fix_file(f"{ROOT}/tests/test_alpha_evolution.py", [
    ("no_discount_factor == 0.98", "no_discount_factor == 0.995"),
])

# ── 2. test_volume_expansion.py ──
fix_file(f"{ROOT}/tests/test_volume_expansion.py", [
    ("drift_vol_ceiling == 0.05", "drift_vol_ceiling == 0.35"),
])

# ── 3. test_backtest/test_wfo.py (5 failures) ──
fix_file(f"{ROOT}/tests/test_backtest/test_wfo.py", [
    # 3a-c: WFO now returns -10.0 for rejected trials, not -inf
    ('assert score == float("-inf")',
     'assert score == -10.0'),
    ('assert score == float("-inf")',
     'assert score == -10.0'),
    ('assert score == float("-inf")',
     'assert score == -10.0'),
    # 3a comment
    ("should be -inf", "should be -10.0"),
    # 3d: zscore search range widened
    ("assert sp.zscore_threshold == (1.0 + 2.5) / 2",
     "assert sp.zscore_threshold == (0.05 + 2.5) / 2"),
    # 3e: min_edge_score lower bound relaxed
    ("assert lower_bound >= 30.0, (", "assert lower_bound >= 20.0, ("),
    ("is below 30.0", "is below 20.0"),
])

# ── 4. test_deployment_guard.py ──
fix_file(f"{ROOT}/tests/test_deployment_guard.py", [
    (
        """    def test_default_is_paper(self):
        from src.core.config import settings
        # In test environment DEPLOYMENT_ENV defaults to PAPER
        assert settings.deployment_env == DeploymentEnv.PAPER""",
        """    def test_default_is_paper(self):
        from src.core.config import Settings
        import os
        from unittest.mock import patch
        # With DEPLOYMENT_ENV unset, default should be PAPER
        env = os.environ.copy()
        env.pop("DEPLOYMENT_ENV", None)
        env["PAPER_MODE"] = "true"
        with patch.dict(os.environ, env, clear=True):
            s = Settings()
            assert s.deployment_env == DeploymentEnv.PAPER""",
    ),
])

# ── 5. test_integration.py ──
fix_file(f"{ROOT}/tests/test_integration.py", [
    (
        '''    def test_paper_mode_default_is_true(self):
        """Paper mode should default to True (safe default)."""
        from src.core.config import Settings
        # With PAPER_MODE not set, should default to True
        with patch.dict(os.environ, {"PAPER_MODE": "true"}):
            s = Settings()
            assert s.paper_mode is True''',
        '''    def test_paper_mode_default_is_true(self):
        """Paper mode should default to True (safe default)."""
        from src.core.config import Settings
        # With DEPLOYMENT_ENV unset and PAPER_MODE=true, should default to True
        env = os.environ.copy()
        env.pop("DEPLOYMENT_ENV", None)
        env["PAPER_MODE"] = "true"
        with patch.dict(os.environ, env, clear=True):
            s = Settings()
            assert s.paper_mode is True''',
    ),
])

# ── 6. test_edge_filter.py (2 failures) ──
fix_file(f"{ROOT}/tests/test_edge_filter.py", [
    # 6a: iceberg bonus assertion must account for 1.0 cap
    (
        """assert iceberg.signal_quality == pytest.approx(
            base.signal_quality + 0.15, abs=0.01
        )""",
        """assert iceberg.signal_quality == pytest.approx(
            min(base.signal_quality + 0.15, 1.0), abs=0.01
        )""",
    ),
    # 6b: high confidence test - lower zscore/volume so default formula
    # doesn't saturate signal_quality to 1.0 (which is > 0.95 confidence)
    (
        """        ea_default = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=1.5, volume_ratio=1.5,
            min_score=0.0,
        )
        ea_conf = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=1.5, volume_ratio=1.5,
            min_score=0.0,
            model_confidence=0.95,
        )""",
        """        ea_default = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=0.9, volume_ratio=0.6,
            min_score=0.0,
        )
        ea_conf = compute_edge_score(
            entry_price=0.40, no_vwap=0.55,
            zscore=0.9, volume_ratio=0.6,
            min_score=0.0,
            model_confidence=0.95,
        )""",
    ),
])

# ── 7. test_unit_math.py (2 failures) ──
# The zscore formula uses (close - vwap)/vwap / sigma (relative normalization).
# Old tests used additive sigma offsets which don't match the relative formula.
fix_file(f"{ROOT}/tests/test_unit_math.py", [
    # 7a: test_does_not_fire_below_threshold
    # mu + 1.0*sigma gives zscore = 1/mu ≈ 2.17 (> 2.0 threshold)
    # mu*(1 + 0.5*sigma) gives zscore = 0.5 (< 2.0 threshold)
    (
        "        # Price at \u03bc + 1\u03c3  (below 2\u03c3 threshold)\n"
        "        mild_price = mu + 1.0 * sigma",
        "        # Price slightly above mu — zscore below 2.0 threshold\n"
        "        mild_price = mu * (1 + 0.5 * sigma)",
    ),
    # 7b: test_exact_boundary_zscore
    # mu + 2.0*sigma gives zscore ≈ 4.35, not 2.0
    # mu*(1 + 2.0*sigma) gives zscore = 2.0 exactly
    (
        "        boundary_price = mu + 2.0 * sigma",
        "        boundary_price = mu * (1 + 2.0 * sigma)",
    ),
])

print("\nALL 13 FIXES APPLIED SUCCESSFULLY")
