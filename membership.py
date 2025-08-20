# membership.py
# Lưu & kiểm tra quyền Plus bằng SQLite (bền vững qua restart)

import os
import sqlite3
from datetime import datetime, timedelta

DB = os.getenv("MEMBERSHIP_DB", "/mnt/data/membership.sqlite3")


def _conn():
    """Kết nối DB và đảm bảo bảng tồn tại."""
    c = sqlite3.connect(DB, detect_types=sqlite3.PARSE_DECLTYPES)
    c.execute(
        "CREATE TABLE IF NOT EXISTS subscriptions("
        "user_id INTEGER PRIMARY KEY, "
        "expires_at TEXT)"
    )
    return c


def activate_plus(user_id: int, days: int = 30) -> datetime:
    """
    Kích hoạt/gia hạn Plus cho user_id thêm {days} ngày.
    Trả về mốc hết hạn (UTC).
    """
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
        c.execute(
            "INSERT INTO subscriptions(user_id, expires_at) VALUES(?,?)",
            (user_id, exp.isoformat()),
        )
    c.commit()
    c.close()
    return exp


def has_plus(user_id: int) -> bool:
    """
    Trả True nếu user đang còn hạn Plus (so với thời điểm hiện tại UTC).
    """
    c = _conn()
    cur = c.execute("SELECT expires_at FROM subscriptions WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    c.close()
    if not row:
        return False
    return datetime.fromisoformat(row[0]) > datetime.utcnow()


def get_expiry(user_id: int):
    """
    Lấy mốc hết hạn Plus (datetime) hoặc None nếu chưa có.
    """
    c = _conn()
    cur = c.execute("SELECT expires_at FROM subscriptions WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    c.close()
    if not row:
        return None
    try:
        return datetime.fromisoformat(row[0])
    except Exception:
        return None


def remaining_days(user_id: int) -> int:
    """
    Số ngày còn lại (làm tròn xuống). Nếu chưa có Plus -> 0.
    """
    exp = get_expiry(user_id)
    if not exp:
        return 0
    delta = exp - datetime.utcnow()
    return max(0, int(delta.total_seconds() // 86400))
