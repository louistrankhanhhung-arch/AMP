# telegram_poster.py
# -*- coding: utf-8 -*-
"""
Đăng signal lên Telegram channel theo 2 chế độ:
- FREE (unmasked) — quota theo ngày
- PLUS (teaser, che số bằng 🔒 + <tg-spoiler>), kèm nút deep-link mở DM bot
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
    signal_id: str
    symbol: str             # "BTCUSDT" (không có '/')
    timeframe: str          # "1H" | "4H" | "1D" ...
    side: str               # "long" | "short" | "none"
    strategy: str           # "trend-follow", "reclaim", ...
    entries: List[float]
    sl: float
    tps: List[float]
    leverage: Optional[int] = None
    eta: Optional[str] = None
    chart_url: Optional[str] = None


# ======================
# 2) Policy: quota FREE / day
# ======================

class DailyQuotaPolicy:
    """
    - Mỗi ngày tối đa `max_free_per_day` post FREE (mặc định 2).
    - Chỉ cho FREE khi đã có ít nhất `min_plus_between_free` bài PLUS kể từ lần FREE gần nhất (mặc định 5).
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
        day, free_c, plus_c, plus_gap = self._roll_day_if_needed()

        if force_free:
            free_c += 1
            plus_gap = 0
            self._save(day, free_c, plus_c, plus_gap)
            return True

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

def _display_symbol(sym: str) -> str:
    if "/" in sym:
        return sym
    # thêm "/" cho các hậu tố phổ biến
    suffixes = ["USDT","USDC","USD","FDUSD","TUSD","BUSD","BTC","ETH"]
    for q in suffixes:
        if sym.endswith(q) and len(sym) > len(q):
            return f"{sym[:-len(q)]}/{q}"
    return sym

def _label_from_tf(tf: str) -> str:
    tfu = (tf or "").upper()
    intradays = {"5M", "15M", "30M", "45M", "1H", "2H"}
    return "INTRADAY" if tfu in intradays else "SWING"

def render_full(sig: Signal) -> str:
    label = _label_from_tf(sig.timeframe)
    sym = _display_symbol(sig.symbol)
    tps_txt = ", ".join(_fmt_price(p) for p in (sig.tps or [])) if sig.tps else "-"
    entries_txt = ", ".join(_fmt_price(p) for p in (sig.entries or [])) if sig.entries else "-"

    lines = [
        f"{sig.side.upper()} | {sym}",
        label,
        "",
        f"Entry: {entries_txt}",
        f"Stop: {_fmt_price(sig.sl)}",
        f"TP: {tps_txt}",
        "",
        f"Strategy: {sig.strategy or '-'}",
    ]
    return "\n".join(lines)


def render_teaser(sig: Signal) -> str:
    lock = "🔒"
    label = _label_from_tf(sig.timeframe)
    sym = _display_symbol(sig.symbol)
    return (
        f"{sig.side.upper()} | {sym}\n"
        f"{label}\n\n"
        f"Entry: <tg-spoiler>{lock}</tg-spoiler>\n"
        f"Stop:  <tg-spoiler>{lock}</tg-spoiler>\n"
        f"TP:    <tg-spoiler>{lock}</tg-spoiler>\n\n"
        f"Strategy: {sig.strategy or '-'}"
    )



# ======================
# 4) Deep-link / post
# ======================

def _bot_username(bot: TeleBot) -> str:
    return bot.get_me().username

def make_deeplink(bot: TeleBot, sig: Signal) -> str:
    return f"https://t.me/{_bot_username(bot)}?start=SIG_{sig.signal_id}"

def make_upgrade_link(bot: TeleBot) -> str:
    # Không có landing page: deep-link về DM để hiển thị Paywall
    return f"https://t.me/{_bot_username(bot)}?start=UPGRADE"

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
    join_btn_url: Optional[str] = None
) -> Dict[str, Any]:
    is_free = policy.decide_is_free(
        max_free_per_day=max_free_per_day,
        min_plus_between_free=min_plus_between_free,
        force_free=force_free,
        ignore_quota=ignore_quota
    )

    kb = types.InlineKeyboardMarkup()

    if is_free:
        text = render_full(sig)
        # Có thể thêm nút nâng cấp nếu muốn (không bắt buộc)
        if join_btn_url:
            kb.add(types.InlineKeyboardButton("✨ Tham gia VIP Membership", url=join_btn_url))
            markup = kb
        else:
            markup = None
    else:
        text = render_teaser(sig)
        deep = make_deeplink(bot, sig)
        kb.add(types.InlineKeyboardButton("🔓 Xem đầy đủ", url=deep))
        upgrade_url = join_btn_url or make_upgrade_link(bot)
        kb.add(types.InlineKeyboardButton("✨ Nâng cấp/Gia hạn Plus", url=upgrade_url))
        markup = kb

    msg = bot.send_message(
        chat_id=channel_id,
        text=text,
        parse_mode="HTML",
        reply_markup=markup,
        disable_web_page_preview=True
    )
    return {"mode": "FREE" if is_free else "PLUS", "chat_id": msg.chat.id, "message_id": msg.message_id}
