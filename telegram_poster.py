# telegram_poster.py
# -*- coding: utf-8 -*-
"""
Đăng signal lên Telegram channel theo 2 chế độ:
- FREE (unmasked) — quota theo ngày
- PLUS (teaser, che số bằng 🔒 + <tg-spoiler>), kèm nút deep-link mở DM bot

Yêu cầu:
- pyTelegramBotAPI (TeleBot)
- Bot đã là admin của channel
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import sqlite3
import datetime as dt

from telebot import TeleBot, types


# ======================
# 1) Mô tả tín hiệu
# ======================

@dataclass
class Signal:
    signal_id: str            # định danh duy nhất, dùng cho deep-link & cache
    symbol: str               # "BTCUSDT" (không có '/')
    timeframe: str            # "1H" | "4H" | "1D" ...
    side: str                 # "long" | "short"
    strategy: str             # "trend-follow", ...
    entries: List[float]
    sl: float
    tps: List[float]
    leverage: Optional[int] = None
    eta: Optional[str] = None  # "1-3d" ...
    chart_url: Optional[str] = None


# ======================
# 2) Policy: quota FREE / day
# ======================

class DailyQuotaPolicy:
    """
    Lưu quota trong SQLite để bền vững qua restart.

    Quy tắc:
    - Mỗi ngày tối đa `max_free_per_day` post FREE (mặc định 2).
    - Chỉ cho FREE khi đã có ít nhất `min_plus_between_free` bài PLUS kể từ lần FREE gần nhất
      (mặc định 5).
    - Có thể ép post FREE qua `force_free=True`.
    """

    def __init__(self, db_path: str = "/mnt/data/policy.sqlite3", key: str = "global"):
        self.db_path = db_path
        self.key = key
        self._ensure()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _ensure(self):
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS policy_state("
                "key TEXT PRIMARY KEY, "
                "day TEXT, "
                "free_count INTEGER, "
                "plus_count INTEGER, "
                "plus_since_last_free INTEGER, "
                "last_post_ts TEXT)"
            )
            cur.execute("SELECT 1 FROM policy_state WHERE key=?", (self.key,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO policy_state(key, day, free_count, plus_count, plus_since_last_free, last_post_ts) "
                    "VALUES(?,?,?,?,?,?)",
                    (self.key, self._today(), 0, 0, 0, dt.datetime.utcnow().isoformat()),
                )
            conn.commit()
        finally:
            conn.close()

    def _today(self) -> str:
        return dt.datetime.utcnow().strftime("%Y-%m-%d")

    def _load(self):
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT day, free_count, plus_count, plus_since_last_free FROM policy_state WHERE key=?",
                (self.key,),
            )
            row = cur.fetchone()
            if not row:
                return self._today(), 0, 0, 0
            return row[0], int(row[1]), int(row[2]), int(row[3])
        finally:
            conn.close()

    def _save(self, day: str, free_count: int, plus_count: int, plus_since_last_free: int):
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE policy_state SET day=?, free_count=?, plus_count=?, plus_since_last_free=?, last_post_ts=? "
                "WHERE key=?",
                (day, free_count, plus_count, plus_since_last_free, dt.datetime.utcnow().isoformat(), self.key),
            )
            conn.commit()
        finally:
            conn.close()

    def _roll_day_if_needed(self):
        day, free_c, plus_c, plus_gap = self._load()
        today = self._today()
        if day != today:
            day, free_c, plus_c, plus_gap = today, 0, 0, 0
            self._save(day, free_c, plus_c, plus_gap)
        return day, free_c, plus_c, plus_gap

    def decide_is_free(
        self,
        *,
        max_free_per_day: int = 2,
        min_plus_between_free: int = 5,
        force_free: bool = False,
        ignore_quota: bool = False
    ) -> bool:
        """
        Trả về True -> bài FREE; False -> bài PLUS.
        Đồng thời cập nhật bộ đếm trong DB.
        """
        day, free_c, plus_c, plus_gap = self._roll_day_if_needed()

        # Nếu ép FREE
        if force_free:
            free_c += 1
            plus_gap = 0
            self._save(day, free_c, plus_c, plus_gap)
            return True

        # Điều kiện FREE theo quota + giãn cách
        if not ignore_quota:
            if free_c >= max_free_per_day:
                plus_c += 1
                plus_gap += 1
                self._save(day, free_c, plus_c, plus_gap)
                return False
            if plus_gap < min_plus_between_free:
                plus_c += 1
                plus_gap += 1
                self._save(day, free_c, plus_c, plus_gap)
                return False

        # Cho FREE
        free_c += 1
        plus_gap = 0
        self._save(day, free_c, plus_c, plus_gap)
        return True


# ======================
# 3) Render nội dung
# ======================

def _fmt_price(x: float) -> str:
    s = f"{x:.8f}".rstrip('0').rstrip('.')
    return s if s else "0"

def _label_from_tf(tf: str) -> str:
    tfu = (tf or "").upper()
    intradays = {"5M", "15M", "30M", "45M", "1H", "2H"}
    return "INTRADAY" if tfu in intradays else "SWING"

def render_full(sig: Signal) -> str:
    tps = "\n".join(f"<b>TP{i+1}:</b> {_fmt_price(p)}" for i, p in enumerate(sig.tps))
    lev = f"\n<b>Leverage:</b> x{sig.leverage}" if sig.leverage else ""
    eta = f"\n<b>ETA:</b> {sig.eta}" if sig.eta else ""
    label = _label_from_tf(sig.timeframe)
    entry_txt = _fmt_price(sig.entries[0]) if (sig.entries and len(sig.entries) > 0) else "-"
    return (
        f"<b>#{sig.symbol}</b> — <b>{sig.side.upper()}</b> {sig.timeframe} ({label})\n"
        f"<b>Entry:</b> {entry_txt}\n"
        f"<b>SL:</b> {_fmt_price(sig.sl)}\n"
        f"{tps}{lev}{eta}"
    )

def render_teaser(sig: Signal) -> str:
    """
    Che số bằng biểu tượng 🔒 và bọc <tg-spoiler> để người dùng Plus mở DM lấy bản full.
    """
    lock = "🔒"
    tps_lock = lock
    label = _label_from_tf(sig.timeframe)
    return (
        f"<b>{sig.symbol} {sig.timeframe} ({label})</b>\n"
        f"Setup: {sig.strategy}\n"
        f"Entry: <tg-spoiler>{lock}</tg-spoiler> | "
        f"SL: <tg-spoiler>{lock}</tg-spoiler> | "
        f"TP: <tg-spoiler>{tps_lock}</tg-spoiler>"
    )


# ======================
# 4) Deep-link / post
# ======================

def make_deeplink(bot: TeleBot, sig: Signal) -> str:
    username = bot.get_me().username
    return f"https://t.me/{username}?start=SIG_{sig.signal_id}"

def post_signal(
    bot: TeleBot,
    channel_id: int,
    sig: Signal,
    policy: DailyQuotaPolicy,
    *,
    max_free_per_day: int = 2,
    min_plus_between_free: int = 5,
    force_free: bool = False,
    ignore_quota: bool = False,
    join_btn_url: Optional[str] = None  # link landing/FAQ thanh toán
) -> Dict[str, Any]:
    """
    - Quyết định FREE/PLUS theo policy hàng ngày.
    - FREE -> gửi full ngay trên channel.
    - PLUS -> gửi teaser + nút deep-link '🔓 Xem đầy đủ', kèm nút upgrade nếu có join_btn_url.
    """
    is_free = policy.decide_is_free(
        max_free_per_day=max_free_per_day,
        min_plus_between_free=min_plus_between_free,
        force_free=force_free,
        ignore_quota=ignore_quota
    )

    kb = types.InlineKeyboardMarkup()

    if is_free:
        text = render_full(sig)
        if join_btn_url:
            kb.add(types.InlineKeyboardButton("✨ Tham gia VIP Membership", url=join_btn_url))
        markup = kb if join_btn_url else None
    else:
        text = render_teaser(sig)
        deep = make_deeplink(bot, sig)
        kb.add(types.InlineKeyboardButton("🔓 Xem đầy đủ", url=deep))
        if join_btn_url:
            kb.add(types.InlineKeyboardButton("✨ Nâng cấp/Gia hạn Plus", url=join_btn_url))
        markup = kb

    msg = bot.send_message(
        chat_id=channel_id,
        text=text,
        parse_mode="HTML",
        reply_markup=markup,
        disable_web_page_preview=True
    )
    return {
        "mode": "FREE" if is_free else "PLUS",
        "chat_id": msg.chat.id,
        "message_id": msg.message_id
    }
