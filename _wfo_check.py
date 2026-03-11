import sqlite3, os, datetime

db = "logs/wfo_phase1.db"
if not os.path.exists(db):
    print(f"DB not found at {db}")
else:
    conn = sqlite3.connect(db)
    c = conn.cursor()
    
    # Inspect schema
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = c.fetchall()
    print(f"Tables: {tables}")
    
    for t in tables:
        tname = t[0]
        c.execute(f"PRAGMA table_info({tname})")
        cols = c.fetchall()
        print(f"\nSchema for {tname}: {cols}")
        c.execute(f"SELECT COUNT(*) FROM {tname}")
        cnt = c.fetchone()[0]
        print(f"Row count: {cnt}")
        if cnt > 0:
            c.execute(f"SELECT * FROM {tname} LIMIT 3")
            rows = c.fetchall()
            print(f"Sample rows: {rows}")
    
    # Trial state counts
    c.execute("SELECT state, COUNT(*) FROM trials GROUP BY state")
    for row in c.fetchall():
        print(f"State '{row[0]}': {row[1]} trials")
    
    # Best sharpe from trial_values
    c.execute("SELECT MAX(tv.value) FROM trial_values tv JOIN trials t ON tv.trial_id=t.trial_id WHERE t.state='COMPLETE'")
    best = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
    complete = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trials")
    total = c.fetchone()[0]
    c.execute("SELECT AVG(tv.value) FROM trial_values tv JOIN trials t ON tv.trial_id=t.trial_id WHERE t.state='COMPLETE'")
    avg = c.fetchone()[0]
    
    # Top 5 sharpes
    c.execute("SELECT tv.value FROM trial_values tv JOIN trials t ON tv.trial_id=t.trial_id WHERE t.state='COMPLETE' ORDER BY tv.value DESC LIMIT 5")
    top5 = c.fetchall()
    
    # Count of -10.0 penalty trials
    c.execute("SELECT COUNT(*) FROM trial_values WHERE value=-10.0")
    penalty = c.fetchone()[0]
    
    # Per-study breakdown
    c.execute("SELECT t.study_id, s.study_name, COUNT(*) FROM trials t JOIN studies s ON t.study_id=s.study_id WHERE t.state='COMPLETE' GROUP BY t.study_id")
    for row in c.fetchall():
        print(f"Study {row[0]} ({row[1]}): {row[2]} complete")
    
    print(f"\nComplete trials: {complete}/{total}")
    print(f"Best Sharpe: {best}")
    print(f"Avg Sharpe: {avg}")
    print(f"Top 5 Sharpes: {top5}")
    print(f"Penalty (-10.0) trials: {penalty}")
    
    conn.close()

# Check pipeline log
log = "logs/pipeline.log"
if os.path.exists(log):
    mtime = os.path.getmtime(log)
    last_mod = datetime.datetime.fromtimestamp(mtime)
    now = datetime.datetime.now()
    delta = (now - last_mod).total_seconds()
    print(f"\nPipeline log last modified: {last_mod}")
    print(f"Seconds since last update: {delta:.0f}")
    if delta > 900:
        print("WARNING: Log stale > 15 minutes!")
    else:
        print("Log is fresh (updated within 15 min)")
    
    # Print last 5 lines
    with open(log) as f:
        lines = f.readlines()
        print(f"\nLast 5 log lines:")
        for l in lines[-5:]:
            print(l.rstrip())
else:
    print(f"Pipeline log not found at {log}")

print(f"\nCurrent time: {datetime.datetime.now()}")
