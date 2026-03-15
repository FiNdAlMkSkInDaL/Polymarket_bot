import sqlite3, datetime

conn = sqlite3.connect("/home/botuser/polymarket-bot/logs/wfo_phase1.db")
c = conn.cursor()

# Total and completed trials
c.execute("SELECT COUNT(*) FROM trials")
total = c.fetchone()[0]

c.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'")
done = c.fetchone()[0]

c.execute("SELECT COUNT(*) FROM trials WHERE state='RUNNING'")
running = c.fetchone()[0]

c.execute("SELECT COUNT(*) FROM trials WHERE state='FAIL'")
failed = c.fetchone()[0]

print(f"Total trials: {total}")
print(f"Completed: {done}")
print(f"Running: {running}")  
print(f"Failed: {failed}")

# Recent completions for timing
c.execute("SELECT datetime_start, datetime_complete FROM trials WHERE state='COMPLETE' ORDER BY trial_id DESC LIMIT 10")
recent = c.fetchall()
print("\nRecent 10 completions:")
for r in recent:
    print(f"  start={r[0]}  end={r[1]}")

# Calculate average trial duration
c.execute("SELECT datetime_start, datetime_complete FROM trials WHERE state='COMPLETE'")
all_trials = c.fetchall()
durations = []
for start, end in all_trials:
    if start and end:
        try:
            s = datetime.datetime.fromisoformat(start)
            e = datetime.datetime.fromisoformat(end)
            durations.append((e - s).total_seconds())
        except:
            pass

if durations:
    avg_dur = sum(durations) / len(durations)
    print(f"\nAvg trial duration: {avg_dur:.1f}s ({avg_dur/60:.1f}min)")
    remaining = 50 - done  # 50 trials per phase
    if remaining > 0:
        # With 2 workers, effective rate is 2x
        eta_seconds = (remaining * avg_dur) / 2
        eta_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=eta_seconds)
        print(f"Remaining trials: {remaining}")
        print(f"ETA with 2 workers: {eta_time.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Time remaining: {eta_seconds/3600:.1f}h")
    else:
        print("Phase 1 complete!")

# Check score distribution
c.execute("SELECT value FROM trial_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE state='COMPLETE') ORDER BY trial_id DESC LIMIT 20")
scores = c.fetchall()
print("\nRecent 20 scores:")
for s in scores:
    print(f"  {s[0]:.4f}")

# Check study info
c.execute("SELECT study_name FROM studies")
studies = c.fetchall()
print(f"\nStudies: {[s[0] for s in studies]}")

conn.close()
