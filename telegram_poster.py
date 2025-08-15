# telegram_poster.py
# -*- coding: utf-8 -*-
"""
Đăng signal lên Telegram channel theo 2 chế độ:
- FREE (unmasked, full thông tin) — mỗi ngày tối đa 'max_free_per_day'
- PLUS (teaser, che số bằng khóa 🔒 + <tg-spoiler>), kèm nút deep-link mở DM bot

YÊU CẦU:
- pip: pyTelegramBotAPI (telebot)
- Bot phải được add làm admin trong channel
- Deep-link xử lý ở Bot: /start SIG_<signal_id> -> kiểm tra membership -> gửi full / paywall

Tác giả: bạn & GPT-5 Thinking
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os                     # <-- thêm
from pathlib import Path      # <-- thêm
import sqlite3
import datetime as dt

from telebot import TeleBot, types

# ======================
# 1) Kiểu dữ liệu Signal
# ======================

@dataclass
class Signal:
    signal_id: str
    symbol: str                # "BTCUSDT"
    timeframe: str             # "1H" | "4H" | "1D" ...
    side: str                  # "long" | "short"
    strategy: str              # "Trend-Follow", ...
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
      (mặc định 5) -> giúp phân tán đều kiểu ~ 2 FREE / ~10 PLUS.
    - Có thể ép post FREE qua `force_free=True` (bỏ qua điều kiện giãn cách, nhưng vẫn tôn trọng quota ngày
      trừ khi `ignore_quota=True`).
    """
    def __init__(self, db_path: str = "policy.sqlite3", key: str = "global"):
        # Bảo đảm thư mục cha tồn tại (hữu ích khi trỏ vào /mnt/data trên Railway)
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.db_path = db_path
        self.key = key
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(self.db_path, check_same_thread=False)
        c.execute("""
        CREATE TABLE IF NOT EXISTS policy_state (
            key TEXT PRIMARY KEY,
            day TEXT,
            free_count INTEGER,
            plus_count INTEGER,
            plus_since_last_free INTEGER,
            last_post_ts TEXT
        )
        """)
        return c

    def _today(self) -> str:
        return dt.date.today().isoformat()

    def _load(self) -> Tuple[str, int, int, int]:
        c = self._conn()
        row = c.execute("SELECT day, free_count, plus_count, plus_since_last_free FROM policy_state WHERE key=?",
                        (self.key,)).fetchone()
        if not row:
            c.execute("INSERT INTO policy_state(key, day, free_count, plus_count, plus_since_last_free, last_post_ts) "
                      "VALUES(?,?,?,?,?,?)",
                      (self.key, self._today(), 0, 0, 0, dt.datetime.utcnow().isoformat()))
            c.commit(); c.close()
            return self._today(), 0, 0, 0
        c.close()
        return row[0], int(row[1]), int(row[2]), int(row[3])

    def _save(self, day: str, free_count: int, plus_count: int, plus_since_last_free: int):
        c = self._conn()
        c.execute("UPDATE policy_state SET day=?, free_count=?, plus_count=?, plus_since_last_free=?, last_post_ts=? "
                  "WHERE key=?",
                  (day, free_count, plus_count, plus_since_last_free, dt.datetime.utcnow().isoformat(), self.key))
        c.commit(); c.close()

    def _roll_day_if_needed(self):
        day, free_c, plus_c, plus_gap = self._load()
        today = self._today()
        if day != today:
            # Reset theo ngày mới
            self._save(today, 0, 0, plus_gap)  # giữ khoảng cách từ hôm qua nếu muốn; thường reset về 0
            return today, 0, 0, plus_gap
        return day, free_c, plus_c, plus_gap

    def decide_is_free(
        self,
        max_free_per_day: int = 2,
        min_plus_between_free: int = 5,
        force_free: bool = False,
        ignore_quota: bool = False
    ) -> bool:
        """
        Trả True -> đăng FREE; False -> đăng PLUS (teaser).
        """
        day, free_c, plus_c, plus_gap = self._roll_day_if_needed()

        if force_free:
            if ignore_quota or free_c < max_free_per_day:
                self._save(day, free_c + 1, plus_c, 0)  # reset gap sau free
                return True
            # Nếu hết quota và không bỏ qua quota: rơi xuống PLUS
            self._save(day, free_c, plus_c + 1, plus_gap + 1)
            return False

        # Tự động: chỉ FREE nếu còn quota và đạt khoảng cách PLUS tối thiểu
        if free_c < max_free_per_day and plus_gap >= min_plus_between_free:
            self._save(day, free_c + 1, plus_c, 0)
            return True

        # Mặc định đăng PLUS
        self._save(day, free_c, plus_c + 1, plus_gap + 1)
        return False


# ======================
# 3) Render nội dung
# ======================

def _fmt_price(x: float) -> str:
    # Bỏ phần thập phân thừa cho gọn (ví dụ 4.7000 -> 4.7)
    s = f"{x:.8f}".rstrip('0').rstrip('.')
    return s if s else "0"

def render_full(sig: Signal) -> str:
    tps = "\n".join(f"<b>TP{i+1}:</b> {_fmt_price(p)}" for i, p in enumerate(sig.tps))
    lev = f"\n<b>Leverage:</b> x{sig.leverage}" if sig.leverage else ""
    eta = f"\n<b>ETA:</b> {sig.eta}" if sig.eta else ""
    return (
        f"<b>#{sig.symbol}</b> — <b>{sig.side.upper()}</b> {sig.timeframe}\n"
        f"<b>Entry:</b> {_fmt_price(sig.entries[0])}\n"
        f"<b>SL:</b> {_fmt_price(sig.sl)}\n"
        f"{tps}{lev}{eta}"
    )

def render_teaser(sig: Signal) -> str:
    """
    Che số bằng biểu tượng 🔒 và bọc <tg-spoiler> để người dùng Plus mở DM lấy bản full.
    """
    lock = "🔒"
    tps_lock = lock  # có thể lặp nhiều: lock * min(3, len(sig.tps))
    return (
        f"<b>{sig.symbol} {sig.timeframe}</b>\n"
        f"Setup: {sig.strategy}\n"
        f"Entry: <tg-spoiler>{lock}</tg-spoiler> | "
        f"SL: <tg-spoiler>{lock}</tg-spoiler> | "
        f"TP: <tg-spoiler>{tps_lock}</tg-spoiler>"
    )

def make_deeplink(bot: TeleBot, sig: Signal) -> str:
    # Deep-link mở DM để bot xử lý quyền xem full theo membership
    username = bot.get_me().username
    return f"https://t.me/{username}?start=SIG_{sig.signal_id}"


# ======================
# 4) Gửi bài lên channel
# ======================

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
    join_btn_url: Optional[str] = None  # ví dụ link đăng ký Plus / hướng dẫn
) -> Dict[str, Any]:
    """
    - Quyết định FREE/PLUS theo policy hàng ngày.
    - FREE -> gửi full ngay trên channel.
    - PLUS -> gửi teaser + nút deep-link '🔓 Xem đầy đủ'.

    Trả về: dict chứa loại bài và (chat_id, message_id).
    """
    is_free = policy.decide_is_free(
        max_free_per_day=max_free_per_day,
        min_plus_between_free=min_plus_between_free,
        force_free=force_free,
        ignore_quota=ignore_quota
    )

    # Kèm ảnh chart nếu có
    markup = None
    text = render_full(sig) if is_free else render_teaser(sig)

    # Nút inline
    kb = types.InlineKeyboardMarkup()
    if is_free:
        # FREE: có thể chèn nút "Tham gia VIP" để upsell
        if join_btn_url:
            kb.add(types.InlineKeyboardButton("✨ Tham gia VIP Membership", url=join_btn_url))
            markup = kb
    else:
        # PLUS: bắt buộc có nút mở DM để xem full
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


# ======================
# 5) Ví dụ sử dụng (tham khảo)
# ======================
if __name__ == "__main__":
    # 1) Tạo bot
    import os
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHANNEL_ID = int(os.getenv("TELEGRAM_CHANNEL_ID", "0"))  # ví dụ: -1001234567890
    JOIN_URL = os.getenv("JOIN_URL", None)  # link trang nâng cấp Plus (tuỳ chọn)

    if not BOT_TOKEN or CHANNEL_ID == 0:
        raise SystemExit("Thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHANNEL_ID")

    bot = TeleBot(BOT_TOKEN, parse_mode=None)  # parse_mode set ở send_message

    # 2) Policy (mặc định 2 FREE/ngày)
    # Ưu tiên Volume trên Railway: /mnt/data/policy.sqlite3
    # Có thể override bằng biến môi trường POLICY_DB
    default_db_path = "/mnt/data/policy.sqlite3"
    db_path = os.getenv("POLICY_DB", default_db_path)
    policy = DailyQuotaPolicy(db_path=db_path,
                              key=os.getenv("POLICY_KEY", "global"))

    # 3) Tạo signal mẫu (thực tế lấy từ builder)
    sig = Signal(
        signal_id="BTC-1D-20250812-001",
        symbol="BTCUSDT",
        timeframe="1D",
        side="long",
        strategy="Trend-Follow (Pullback hợp lệ)",
        entries=[67600.0, 66800.0],
        sl=65200.0,
        tps=[69000.0, 70500.0, 72000.0],
        leverage=5,
        eta="1–3d",
        chart_url=None
    )

    # 4) Đăng
    info = post_signal(
        bot=bot,
        channel_id=CHANNEL_ID,
        sig=sig,
        policy=policy,
        max_free_per_day=2,         # quota FREE/ngày
        min_plus_between_free=5,    # giãn cách tối thiểu giữa các FREE
        force_free=False,           # có thể bật True khi cần "mồi" có chủ đích
        join_btn_url=JOIN_URL
    )
    print(info)
