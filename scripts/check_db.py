import sqlite3
db = sqlite3.connect(r"C:\vps_dump\trades.db")
cur = db.cursor()
rows = cur.execute("SELECT name, type FROM sqlite_master").fetchall()
print("All objects in DB:", rows)
for name, typ in rows:
    if typ == "table":
        count = cur.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
        print(f"  {name}: {count} rows")
        # Show columns
        cols = cur.execute(f"PRAGMA table_info([{name}])").fetchall()
        print(f"  Columns: {[c[1] for c in cols]}")
db.close()
