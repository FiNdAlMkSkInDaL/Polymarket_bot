#!/usr/bin/env python3
"""Verify WFO trial objectives after numeric penalty fix."""
import sqlite3, os, glob

LOG_DIR = "/home/botuser/polymarket-bot/logs"

# Find the active WFO DB
dbs = sorted(glob.glob(os.path.join(LOG_DIR, "wfo*.db")))
print(f"WFO databases found: {[os.path.basename(d) for d in dbs]}")

for db_path in dbs:
    print(f"\n{'='*50}")
    print(f"DB: {os.path.basename(db_path)}")
    db = sqlite3.connect(db_path)
    cur = db.cursor()

    # Check tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    if "studies" not in tables:
        print(f"  Not an Optuna DB (tables: {tables})")
        db.close()
        continue

    cur.execute("SELECT study_id, study_name FROM studies")
    studies = cur.fetchall()

    for study_id, study_name in studies:
        print(f"\n  Study: {study_name}")

        # Trial states
        cur.execute("SELECT state, COUNT(*) FROM trials WHERE study_id=? GROUP BY state", (study_id,))
        states = dict(cur.fetchall())
        total = sum(states.values())
        print(f"  States: {states} (total={total})")

        # All trial values for COMPLETE trials
        cur.execute("""
            SELECT t.trial_id, t.number, t.state, tv.value
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ?
            ORDER BY t.number
        """, (study_id,))
        rows = cur.fetchall()

        null_count = sum(1 for r in rows if r[3] is None)
        neg10_count = sum(1 for r in rows if r[3] == -10.0)
        valid_count = sum(1 for r in rows if r[3] is not None and r[3] > -10.0)

        print(f"  NULL objectives: {null_count}")
        print(f"  -10.0 penalty objectives: {neg10_count}")
        print(f"  Valid positive objectives: {valid_count}")

        # Show first 10 rows
        print(f"\n  First 10 trials:")
        print(f"  {'trial_id':>8} {'number':>6} {'state':>10} {'value':>12}")
        for r in rows[:10]:
            val_str = f"{r[3]:.4f}" if r[3] is not None else "NULL"
            print(f"  {r[0]:>8} {r[1]:>6} {r[2]:>10} {val_str:>12}")

        # Top 3 best
        cur.execute("""
            SELECT tv.value FROM trial_values tv
            JOIN trials t ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND tv.value > -10.0
            ORDER BY tv.value DESC LIMIT 3
        """, (study_id,))
        top = cur.fetchall()
        if top:
            print(f"\n  Top 3 best objectives: {[round(v[0], 4) for v in top]}")

    db.close()

# Also check log tail
print(f"\n{'='*50}")
print("WFO_FRESH.LOG LAST 10 LINES:")
fresh = os.path.join(LOG_DIR, "wfo_fresh.log")
if os.path.exists(fresh):
    with open(fresh) as f:
        lines = f.readlines()
    print(f"  Total lines: {len(lines)}")
    for l in lines[-10:]:
        print(f"  {l.strip()[:180]}")

# Check tmux is still alive
import subprocess
r = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True)
print(f"\nTmux sessions: {r.stdout.strip()}")

# Process check
r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
for line in r.stdout.split('\n'):
    if 'run_optimization' in line and 'grep' not in line:
        print(f"WFO process: {line.strip()[:150]}")
