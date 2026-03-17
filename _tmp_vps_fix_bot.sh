#!/usr/bin/env bash
set -euo pipefail
cd /home/botuser/polymarket-bot

git checkout -- src/bot.py

python3 - <<'PY'
from pathlib import Path
p = Path('src/bot.py')
text = p.read_text(encoding='utf-8')
old = '''                mc = OracleMarketConfig(
                    market_id=cfg_dict.get("market_id", ""),
                    oracle_type=cfg_dict.get("oracle_type", ""),
                    oracle_params=cfg_dict.get("oracle_params", {}),
                    yes_asset_id=cfg_dict.get("yes_asset_id", ""),
                    no_asset_id=cfg_dict.get("no_asset_id", ""),
                    event_id=cfg_dict.get("event_id", ""),
                )'''
new = '''                mc = OracleMarketConfig(
                    market_id=cfg_dict.get("market_id", ""),
                    oracle_type=cfg_dict.get("oracle_type", ""),
                    oracle_params=cfg_dict.get("oracle_params", {}),
                    external_id=cfg_dict.get(
                        "external_id",
                        cfg_dict.get("oracle_params", {}).get("match_id", ""),
                    ),
                    target_outcome=cfg_dict.get(
                        "target_outcome",
                        cfg_dict.get("oracle_params", {}).get("team", ""),
                    ),
                    market_type=cfg_dict.get(
                        "market_type",
                        cfg_dict.get("oracle_params", {}).get("market_type", "winner"),
                    ),
                    goal_line=float(
                        cfg_dict.get(
                            "goal_line",
                            cfg_dict.get("oracle_params", {}).get("goal_line", 2.5),
                        )
                    ),
                    yes_asset_id=cfg_dict.get("yes_asset_id", ""),
                    no_asset_id=cfg_dict.get("no_asset_id", ""),
                    event_id=cfg_dict.get("event_id", ""),
                )'''
if old not in text:
    raise SystemExit('expected parser block not found in src/bot.py')
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
PY

grep -n "external_id=cfg_dict.get\|target_outcome=cfg_dict.get\|goal_line=float" src/bot.py
