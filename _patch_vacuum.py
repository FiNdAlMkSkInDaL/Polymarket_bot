"""Patch position_manager.py to add spread-aware clamping for vacuum stink bids."""

filepath = '/home/botuser/polymarket-bot/src/trading/position_manager.py'

with open(filepath, 'r') as f:
    content = f.read()

old_block = (
    "        # mid is the YES token mid-price; NO mid = 1.0 - mid.\n"
    "        # Each stink bid sits *below* that side's mid to catch flash-crash wicks.\n"
    "        yes_bid = round(max(0.01, mid - offset), 2)\n"
    "        no_bid = round(max(0.01, (1.0 - mid) - offset), 2)\n"
    "\n"
    "        sides: list[tuple[float, str, str, float]] = []\n"
    "        # BUY YES token below YES mid \u2014 catches YES flash-crash\n"
    "        if 0.01 <= yes_bid <= 0.99 and signal.yes_asset_id:\n"
    "            sides.append((yes_bid, signal.yes_asset_id, \"BUY\", mid))\n"
    "        # BUY NO token below NO mid \u2014 catches NO flash-crash\n"
    "        if 0.01 <= no_bid <= 0.99 and signal.no_asset_id:\n"
    "            sides.append((no_bid, signal.no_asset_id, \"BUY\", 1.0 - mid))"
)

new_block = (
    "        # mid is the YES token mid-price; NO mid = 1.0 - mid.\n"
    "        # Each stink bid sits *below* that side's mid to catch flash-crash wicks.\n"
    "        yes_bid = round(max(0.01, mid - offset), 2)\n"
    "        no_bid = round(max(0.01, (1.0 - mid) - offset), 2)\n"
    "\n"
    "        # \u2500\u2500 Spread-aware clamping \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "        # During ghost-liquidity events the spread blows out.  If the\n"
    "        # calculated bid >= best_ask, POST_ONLY would reject because the\n"
    "        # order would cross.  Clamp each bid to best_ask - 0.01 so it\n"
    "        # always rests as a maker order.\n"
    "        yes_tracker = self._book_trackers.get(signal.yes_asset_id)\n"
    "        no_tracker = self._book_trackers.get(signal.no_asset_id)\n"
    "\n"
    "        if yes_tracker is not None and yes_tracker.best_ask > 0:\n"
    "            yes_bid = round(min(yes_bid, yes_tracker.best_ask - 0.01), 2)\n"
    "        if no_tracker is not None and no_tracker.best_ask > 0:\n"
    "            no_bid = round(min(no_bid, no_tracker.best_ask - 0.01), 2)\n"
    "\n"
    "        sides: list[tuple[float, str, str, float]] = []\n"
    "        # BUY YES token below YES mid \u2014 catches YES flash-crash\n"
    "        if 0.01 <= yes_bid <= 0.99 and signal.yes_asset_id:\n"
    "            sides.append((yes_bid, signal.yes_asset_id, \"BUY\", mid))\n"
    "        # BUY NO token below NO mid \u2014 catches NO flash-crash\n"
    "        if 0.01 <= no_bid <= 0.99 and signal.no_asset_id:\n"
    "            sides.append((no_bid, signal.no_asset_id, \"BUY\", 1.0 - mid))"
)

if old_block not in content:
    print('ERROR: old block not found in file!')
    # Debug: check each line
    for i, line in enumerate(old_block.split('\n')):
        if line not in content:
            print(f'  Line {i} not found: {line!r}')
else:
    content = content.replace(old_block, new_block, 1)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCH APPLIED SUCCESSFULLY')

    # Verify
    with open(filepath, 'r') as f:
        verify = f.read()
    if 'Spread-aware clamping' in verify and 'yes_tracker.best_ask - 0.01' in verify:
        print('VERIFIED: New clamping logic is present')
    else:
        print('ERROR: Verification failed')
