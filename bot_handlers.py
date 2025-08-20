# bot_handlers.py
from __future__ import annotations

import os, re, logging
from datetime import datetime
from typing import Set

from telebot import TeleBot, types

# Membership core
from app.membership import (
    has_plus,
    activate_plus,
    get_expiry,
    remaining_days,
)

# Nội dung signal để mở khóa trong DM
from app.signal_tracker import render_full_signal_by_id
# Tóm tắt/teaser khi user chưa Plus (tùy chọn, nếu không có sẽ fallback)
try:
    from app.performance_logger import summarize_signal
except Exception:
    summarize_signal = None  # type: ignore


# =============== Helpers ===============

def _admin_ids() -> Set[int]:
    s = os.getenv("ADMIN_IDS", "")
    return {int(x) for x in re.findall(r"\d+", s)}

def _is_admin(user_id: int) -> bool:
    return user_id in _admin_ids()

def _plus_link(bot: TeleBot) -> str:
    # Ưu tiên trang thanh toán riêng nếu có
    join = os.getenv("JOIN_URL")
    if join:
        return join
    # Mặc định deeplink về DM để hiển thị Paywall
    uname = bot.get_me().username
    return f"https://t.me/{uname}?start=UPGRADE}"

def _paywall_text() -> str:
    txt = os.getenv("PAYWALL_TEXT")
    if txt:
        return txt
    # Mặc định: hướng dẫn chuyển khoản ngắn gọn
    return (
        "✨ <b>Nâng cấp/gia hạn gói Plus</b>\n"
        "• Chuyển khoản theo hướng dẫn tại trang Paywall.\n"
        "• Sau khi chuyển xong, bấm nút <b>“Đã chuyển tiền”</b> để báo admin.\n"
        "• Admin sẽ kích hoạt trong thời gian sớm nhất.\n"
    )

def _format_status(uid: int) -> str:
    days = remaining_days(uid)
    exp = get_expiry(uid)
    if days <= 0 or not exp:
        return "📦 <b>Gói hiện tại:</b> Free\n⏳ <i>Chưa có hạn Plus</i>"
    # hiển thị theo UTC để nhất quán
    return (
        f"📦 <b>Gói hiện tại:</b> Plus\n"
        f"⏳ <b>Còn lại:</b> {days} ngày\n"
        f"🗓 <b>Hết hạn (UTC):</b> {exp.strftime('%Y-%m-%d %H:%M:%S')}"
    )


# =============== Register ===============

def register_handlers(bot: TeleBot):
    # /start [TOKEN]
    @bot.message_handler(commands=['start'])
    def on_start(m):
        token = None
        parts = m.text.split(maxsplit=1)
        if len(parts) > 1:
            token = parts[1].strip()

        # 1) Deeplink UPGRADE -> gửi Paywall + nút "Đã chuyển tiền"
        if token == "UPGRADE":
            kb = types.InlineKeyboardMarkup()
            kb.add(types.InlineKeyboardButton("✅ Đã chuyển tiền", callback_data=f"PAID_CONFIRMED:{m.from_user.id}"))
            bot.send_message(
                m.chat.id,
                _paywall_text(),
                parse_mode='HTML',
                reply_markup=kb,
                disable_web_page_preview=True
            )
            return

        # 2) Deeplink SIG_<id> -> unlock nếu có Plus, ngược lại gửi teaser + link nâng cấp
        if token and token.startswith('SIG_'):
            sig_id = token[4:]
            # Khuyến nghị: chỉ gửi full trong DM
            if m.chat.type != 'private':
                link = _plus_link(bot)
                bot.send_message(
                    m.chat.id,
                    f"Vui lòng mở DM với bot để xem chi tiết.\nLink: {link}",
                    disable_web_page_preview=True
                )
                return

            if has_plus(m.from_user.id):
                try:
                    html = render_full_signal_by_id(sig_id)
                    if not html:
                        raise ValueError("signal not found")
                    bot.send_message(m.chat.id, html, parse_mode='HTML', disable_web_page_preview=True)
                except Exception as e:
                    logging.exception(f"render_full_signal_by_id error: {e}")
                    bot.send_message(m.chat.id, "Xin lỗi, không tìm thấy signal hoặc xảy ra lỗi khi tải.")
            else:
                # Chưa Plus: gửi teaser (nếu có), kèm nút nâng cấp
                kb = types.InlineKeyboardMarkup()
                kb.add(types.InlineKeyboardButton("✨ Nâng cấp/Gia hạn Plus", url=_plus_link(bot)))
                teaser = None
                if summarize_signal:
                    try:
                        teaser = summarize_signal(sig_id)
                    except Exception:
                        teaser = None
                bot.send_message(
                    m.chat.id,
                    teaser or "Bạn đang ở gói Free. Một phần nội dung đã bị ẩn.",
                    parse_mode='HTML',
                    reply_markup=kb,
                    disable_web_page_preview=True
                )
            return

        # 3) Không có token -> chào mừng + gợi ý lệnh
        bot.send_message(
            m.chat.id,
            "Chào mừng! Bạn có thể dùng:\n"
            "• /status — xem trạng thái gói Plus\n"
            "• /plus_link — nhận link nâng cấp/gia hạn\n",
            parse_mode='HTML'
        )

    # /status — trạng thái gói
    @bot.message_handler(commands=['status'])
    def on_status(m):
        bot.send_message(m.chat.id, _format_status(m.from_user.id), parse_mode='HTML')

    # /plus_link — gửi link nâng cấp/gia hạn
    @bot.message_handler(commands=['plus_link'])
    def on_plus_link(m):
        link = _plus_link(bot)
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton("✨ Nâng cấp/Gia hạn Plus", url=link))
        bot.send_message(m.chat.id, f"Link nâng cấp/gia hạn:\n{link}", reply_markup=kb, disable_web_page_preview=True)

    # /plus_add <user_id> <days> — admin
    @bot.message_handler(commands=['plus_add'])
    def on_plus_add(m):
        if not _is_admin(m.from_user.id):
            bot.reply_to(m, "Bạn không có quyền thực hiện lệnh này.")
            return
        try:
            _, uid_str, days_str = m.text.strip().split(maxsplit=2)
            uid = int(uid_str)
            days = int(days_str)
            exp = activate_plus(uid, days)
            bot.reply_to(m, f"✅ Đã cộng {days} ngày cho {uid}. Hết hạn mới (UTC): {exp.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            bot.reply_to(m, f"❌ Sai cú pháp. Dùng: /plus_add <user_id> <days>\nErr: {e}")

    # /remove <user_id> — admin (hủy ngay bằng cách set 0 ngày)
    @bot.message_handler(commands=['remove'])
    def on_remove(m):
        if not _is_admin(m.from_user.id):
            bot.reply_to(m, "Bạn không có quyền thực hiện lệnh này.")
            return
        try:
            _, uid_str = m.text.strip().split(maxsplit=1)
            uid = int(uid_str)
            # membership.py chưa có revoke; dùng activate_plus(..., 0) để hết hạn "ngay"
            activate_plus(uid, 0)
            bot.reply_to(m, f"✅ Đã hủy gói Plus của {uid}.")
        except Exception as e:
            bot.reply_to(m, f"❌ Sai cú pháp. Dùng: /remove <user_id>\nErr: {e}")

    # Callback: “Đã chuyển tiền”
    @bot.callback_query_handler(func=lambda c: c.data and c.data.startswith("PAID_CONFIRMED"))
    def on_paid_confirmed(call):
        try:
            parts = call.data.split(":", 1)
            paid_uid = int(parts[1]) if len(parts) > 1 else call.from_user.id
        except Exception:
            paid_uid = call.from_user.id

        bot.answer_callback_query(call.id, "Cảm ơn! Admin sẽ kích hoạt sớm.")
        # Báo cho admin
        text = (
            "💳 <b>Yêu cầu kích hoạt Plus</b>\n"
            f"• user_id: <code>{paid_uid}</code>\n"
            "• Lệnh gợi ý: <code>/plus_add {uid} 30</code>\n".format(uid=paid_uid)
        )
        for aid in _admin_ids():
            try:
                bot.send_message(aid, text, parse_mode='HTML')
            except Exception:
                pass
