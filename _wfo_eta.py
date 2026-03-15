import sqlite3, datetime

conn = sqlite3.connect("/home/botuser/polymarket-bot/logs/wfo_phase1.db")
c = conn.cursor()

# Trials per study/fold
c.execute("""
    SELECT s.study_name, 
           COUNT(t.trial_id) as total,
           SUM(CASE WHEN t.state='COMPLETE' THEN 1 ELSE 0 END) as done,
           SUM(CASE WHEN t.state='RUNNING' THEN 1 ELSE 0 END) as running
    FROM studies s
    LEFT JOIN trials t ON s.study_id = t.study_id
    GROUP BY s.study_name
    ORDER BY s.study_name
""")
folds = c.fetchall()
print("=== TRIALS PER FOLD ===")
for name, total, done, running in folds:
    print(f"  {name}: {done}/{total} complete, {running} running")

# Overall timing
c.execute("""
    SELECT MIN(datetime_start), MAX(datetime_complete)
    FROM trials WHERE state='COMPLETE'
""")
first_start, last_end = c.fetchone()
print(f"\nFirst trial start: {first_start}")
print(f"Last trial end:    {last_end}")

# Phase 2 DB exists?
import os
p2_exists = os.path.exists("/home/botuser/polymarket-bot/logs/wfo_phase2.db")
print(f"\nPhase 2 DB exists: {p2_exists}")

# Current UTC time
now = datetime.datetime.utcnow()
print(f"Current UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}")

# Estimate remaining Phase 1 time
total_phase1_trials = sum(t for _, t, _, _ in folds)
done_phase1 = sum(d for _, _, d, _ in folds)
remaining_phase1 = total_phase1_trials - done_phase1

# Avg duration from recent trials
c.execute("""
    SELECT datetime_start, datetime_complete 
    FROM trials WHERE state='COMPLETE' 
    ORDER BY trial_id DESC LIMIT 20
""")
recent = c.fetchall()
durations = []
for s, e in recent:
    if s and e:
        start = datetime.datetime.fromisoformat(s)
        end = datetime.datetime.fromisoformat(e)
        durations.append((end - start).total_seconds())

avg_dur = sum(durations) / len(durations) if durations else 0
print(f"\nAvg trial duration (last 20): {avg_dur:.1f}s")
print(f"Phase 1 remaining: {remaining_phase1} trials")

# With 2 workers
phase1_eta_s = (remaining_phase1 * avg_dur) / 2
print(f"Phase 1 time remaining: {phase1_eta_s/60:.1f} min")

# Phase 2 estimate: 100 trials * 5 folds = 500 trials
# Phase 2 uses 10 markets & 30 days vs Phase 1's 5 markets & 14 days
# Scale factor: (10/5) * (30/14) ~= 4.3x
scale = (10/5) * (30/14)
phase2_trial_est = avg_dur * scale
phase2_total_trials = 100 * 5  # 100 per fold, 5 folds
phase2_eta_s = (phase2_total_trials * phase2_trial_est) / 2
print(f"\nPhase 2 estimate:")
print(f"  Scale factor: {scale:.1f}x")
print(f"  Est trial duration: {phase2_trial_est:.1f}s")
print(f"  Total trials: {phase2_total_trials}")
print(f"  Time estimate: {phase2_eta_s/3600:.1f}h")

# Phase 3: 1 trial, 31 markets, ~instant
total_eta_s = phase1_eta_s + phase2_eta_s + 300  # 5 min for phase 3
final_eta = now + datetime.timedelta(seconds=total_eta_s)
print(f"\n=== TOTAL WFO ETA ===")
print(f"Estimated completion: {final_eta.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"Total remaining: {total_eta_s/3600:.1f}h")

conn.close()
