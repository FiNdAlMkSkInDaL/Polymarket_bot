"""WFO Pipeline Diagnostic — run on VPS."""
import sqlite3, os, json, time
from datetime import datetime, timezone, timedelta

LOG_DIR = "/home/botuser/polymarket-bot/logs"

def query_db(db_path):
    """Query an Optuna SQLite DB for trial states and timing."""
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Trial states: COMPLETE=1, RUNNING=2 (or similar Optuna encoding)
    # Optuna stores state as text in newer versions
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    print(f"  Tables: {tables}")

    # Get trial info
    try:
        cur.execute("SELECT state, COUNT(*) FROM trials GROUP BY state;")
        states = dict(cur.fetchall())
        print(f"  States: {states}")
    except Exception as e:
        print(f"  State query error: {e}")
        states = {}

    # Get objective values for completed trials
    try:
        cur.execute("""
            SELECT t.trial_id, t.state, tv.value, t.datetime_start, t.datetime_complete
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            ORDER BY t.trial_id;
        """)
        trials = cur.fetchall()
        print(f"  Total trial rows: {len(trials)}")

        completed = [(tid, st, val, ds, dc) for tid, st, val, ds, dc in trials if st == "COMPLETE"]
        running = [(tid, st, val, ds, dc) for tid, st, val, ds, dc in trials if st == "RUNNING"]
        failed = [(tid, st, val, ds, dc) for tid, st, val, ds, dc in trials if st == "FAIL"]

        print(f"  COMPLETE: {len(completed)}, RUNNING: {len(running)}, FAIL: {len(failed)}")

        if completed:
            values = [v for _, _, v, _, _ in completed if v is not None]
            if values:
                print(f"  Objective values: min={min(values):.4f}, max={max(values):.4f}, mean={sum(values)/len(values):.4f}")
                penalty_count = sum(1 for v in values if v == -10.0)
                positive_count = sum(1 for v in values if v > 0)
                print(f"  Penalty (-10.0) count: {penalty_count}")
                print(f"  Positive Sharpe count: {positive_count}")
                best_5 = sorted(values, reverse=True)[:5]
                print(f"  Top 5 objectives: {[round(v, 4) for v in best_5]}")

            # Timing analysis
            durations = []
            for tid, st, val, ds, dc in completed:
                if ds and dc:
                    try:
                        t_start = datetime.fromisoformat(ds.replace("Z", "+00:00"))
                        t_end = datetime.fromisoformat(dc.replace("Z", "+00:00"))
                        dur = (t_end - t_start).total_seconds()
                        durations.append(dur)
                    except Exception:
                        pass

            if durations:
                avg_dur = sum(durations) / len(durations)
                print(f"  Avg trial duration: {avg_dur:.1f}s ({avg_dur/60:.1f}min)")
                print(f"  Min trial duration: {min(durations):.1f}s")
                print(f"  Max trial duration: {max(durations):.1f}s")
                print(f"  Last 5 durations: {[round(d, 1) for d in durations[-5:]]}")

        # Check running trials
        if running:
            for tid, st, val, ds, dc in running:
                print(f"  RUNNING trial {tid}: started={ds}")

    except Exception as e:
        print(f"  Trial detail error: {e}")
        import traceback; traceback.print_exc()

    conn.close()
    return states

print("=" * 60)
print("WFO PIPELINE DIAGNOSTIC")
print("=" * 60)
print(f"Time: {datetime.now(timezone.utc).isoformat()}")
print()

for phase in [1, 2, 3]:
    db_path = f"{LOG_DIR}/wfo_phase{phase}.db"
    print(f"--- Phase {phase} ({db_path}) ---")
    if os.path.exists(db_path):
        sz = os.path.getsize(db_path)
        print(f"  Size: {sz} bytes ({sz/1024:.1f} KB)")
        query_db(db_path)
    else:
        print("  NOT FOUND")
    print()

# Check WFO log
print("--- WFO Log Tail ---")
wfo_log = f"{LOG_DIR}/wfo_fresh.log"
if os.path.exists(wfo_log):
    sz = os.path.getsize(wfo_log)
    print(f"  Log size: {sz} bytes ({sz/1024/1024:.1f} MB)")
    with open(wfo_log, "r") as f:
        lines = f.readlines()
    print(f"  Total lines: {len(lines)}")
    # Last 30 lines
    print("  --- Last 30 lines ---")
    for line in lines[-30:]:
        print(f"  {line.rstrip()}")
else:
    print("  NOT FOUND")

# Check running processes
print()
print("--- Running WFO Processes ---")
os.system("ps aux | grep -E 'run_optimization|wfo|optuna' | grep -v grep")
print()
print("--- tmux sessions ---")
os.system("tmux list-sessions 2>&1")
