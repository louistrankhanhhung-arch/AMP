import os
import sqlite3
import datetime as dt
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
DB_FILE = os.getenv("POST_DB_FILE", "post_state.db")


class DailyQuotaPolicy:
    def __init__(self, key: str):
        self.key = key
        self.db_file = DB_FILE
        self._ensure_table()
        self._init_db()  # ✅ Gọi hàm khởi tạo DB

    def _conn(self):
        return sqlite3.connect(self.db_file)

    def _ensure_table(self):
        c = self._conn()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_state (
                key TEXT PRIMARY KEY,
                day TEXT,
                free_count INTEGER,
                plus_count INTEGER,
                plus_since_last_free INTEGER,
                last_post_ts TEXT
            )
            """
        )
        c.commit()
        c.close()

    def _init_db(self):
        """
        Khởi tạo DB nếu chưa tồn tại record cho key.
        """
        c = self._conn()
        row = c.execute("SELECT key FROM policy_state WHERE key=?", (self.key,)).fetchone()
        if not row:
            c.execute(
                "INSERT INTO policy_state(key, day, free_count, plus_count, plus_since_last_free, last_post_ts) VALUES(?,?,?,?,?,?)",
                (
                    self.key,
                    self._today(),
                    0,
                    0,
                    0,
                    dt.datetime.utcnow().isoformat(),
                ),
            )
            c.commit()
        c.close()

    def _today(self):
        return dt.date.today().isoformat()

    def can_post(self, plus: bool = False) -> bool:
        c = self._conn()
        row = c.execute("SELECT day, free_count, plus_count FROM policy_state WHERE key=?", (self.key,)).fetchone()
        if not row:
            c.close()
            return True

        day, free_count, plus_count = row
        today = self._today()

        if day != today:
            c.execute(
                "UPDATE policy_state SET day=?, free_count=0, plus_count=0, plus_since_last_free=0 WHERE key=?",
                (today, self.key),
            )
            c.commit()
            c.close()
            return True

        c.close()

        if plus:
            return plus_count < 10
        return free_count < 5

    def register_post(self, plus: bool = False):
        c = self._conn()
        today = self._today()
        if plus:
            c.execute(
                "UPDATE policy_state SET plus_count=plus_count+1, last_post_ts=? WHERE key=?",
                (dt.datetime.utcnow().isoformat(), self.key),
            )
        else:
            c.execute(
                "UPDATE policy_state SET free_count=free_count+1, plus_since_last_free=0, last_post_ts=? WHERE key=?",
                (dt.datetime.utcnow().isoformat(), self.key),
            )
        c.commit()
        c.close()


class TelegramPoster:
    def __init__(self):
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHANNEL_ID
        self.policy = DailyQuotaPolicy("telegram")

    def post_signal(self, text: str, plus: bool = False):
        if not self.policy.can_post(plus=plus):
            print("Quota exceeded, not posting")
            return None

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            r = requests.post(url, data=payload, timeout=10)
            if r.status_code == 200:
                self.policy.register_post(plus=plus)
                print("Posted to Telegram")
                return r.json()
            else:
                print("Failed to post:", r.text)
                return None
        except Exception as e:
            print("Error posting to Telegram:", e)
            return None
