import sqlite3
conn = sqlite3.connect('/home/botuser/polymarket-bot/logs/wfo_phase1.db')
c = conn.cursor()

# List tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in c.fetchall()]
print("Tables:", tables)

# Count trials
for tn in tables:
    if 'trial' in tn.lower():
        c.execute(f"SELECT COUNT(*) FROM [{tn}]")
        cnt = c.fetchone()[0]
        print(f"  {tn}: {cnt} rows")

# Check Optuna studies
if 'studies' in tables:
    c.execute("SELECT study_id, study_name FROM studies")
    for row in c.fetchall():
        print(f"  Study: {row}")

# Count completed trials via Optuna's trial_values
if 'trial_values' in tables:
    c.execute("SELECT COUNT(*) FROM trial_values")
    print(f"  trial_values: {c.fetchone()[0]} rows")

# Check trial states
if 'trials' in tables:
    c.execute("SELECT state, COUNT(*) FROM trials GROUP BY state")
    for row in c.fetchall():
        state_name = {0: 'RUNNING', 1: 'COMPLETE', 2: 'PRUNED', 3: 'FAIL', 4: 'WAITING'}.get(row[0], f'UNKNOWN({row[0]})')
        print(f"  State {state_name}: {row[1]} trials")

    # Check last few trial values
    c.execute("SELECT t.trial_id, tv.value FROM trials t LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id ORDER BY t.trial_id DESC LIMIT 10")
    print("  Last 10 trials (id, score):")
    for row in c.fetchall():
        print(f"    Trial {row[0]}: {row[1]}")

conn.close()
