import sqlite3
from datetime import datetime, timedelta

DB = ":memory:"  # replace with file path in production

def _conn():
    c = sqlite3.connect(DB, detect_types=sqlite3.PARSE_DECLTYPES)
    c.execute("CREATE TABLE IF NOT EXISTS subscriptions(user_id INTEGER PRIMARY KEY, expires_at TEXT)")
    return c

def activate_plus(user_id: int, days: int = 30):
    c = _conn()
    now = datetime.utcnow()
    cur = c.execute("SELECT expires_at FROM subscriptions WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if row:
        old = datetime.fromisoformat(row[0])
        base = old if old > now else now
        exp = base + timedelta(days=days)
        c.execute("UPDATE subscriptions SET expires_at=? WHERE user_id=?", (exp.isoformat(), user_id))
    else:
        exp = now + timedelta(days=days)
        c.execute("INSERT INTO subscriptions(user_id, expires_at) VALUES(?,?)", (user_id, exp.isoformat()))
    c.commit(); c.close()
    return exp

def has_plus(user_id: int) -> bool:
    c = _conn()
    cur = c.execute("SELECT expires_at FROM subscriptions WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    c.close()
    if not row: return False
    return datetime.fromisoformat(row[0]) > datetime.utcnow()
