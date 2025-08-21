# bot_handlers.py
from __future__ import annotations

import os, re, logging, random, string
from datetime import datetime, timezone
from typing import Set

from telebot import TeleBot, types

# Membership core
from membership import (
    has_plus,
    activate_plus,
    get_expiry,
    remaining_days,
)

# Nội dung signal để mở khóa trong DM
from signal_tracker import render_full_signal_by_id
# Tóm tắt/teaser khi user chưa Plus (tùy chọn, nếu không có sẽ fallback)
try:
    from performance_logger import summarize_signal
except Exception:
    summarize_signal = None  # type: ignore


# =============== Helpers ===============

def _admin_ids() -> Set[int]:
    s = os.getenv("ADMIN_IDS", "")
    return {int(x) for x in re.findall(r"\d+", s)}

def _is_admin(user_id: int) -> bool:
    return user_id in _admin_ids()

def _gen_order_code(uid: int) -> str:
    # Ví dụ: ORD-YYMMDD-<2 ký tự ngẫu nhiên>
    ts = datetime.now(timezone.utc).strftime("%y%m%d-%H%M")
    rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
    return f"ORD-{ts}-{rand}"

def _render_paywall_html(user_id: int, order_code: str) -> str:
    # Paywall động kèm thông tin ngân hàng + nội dung CK: <mã order> <user_id>
    return (
        "✨ <b>Nâng cấp / Gia hạn gói Plus</b>\n\n"
        "<b>1) Chuyển khoản ngân hàng</b>\n"
        "• <b>Ngân hàng:</b> Ngân hàng Quân đội (MBBank)\n"
        "• <b>Số tài khoản:</b> 0378285345\n"
        "• <b>Chủ tài khoản:</b> Trần Khánh Hưng\n\n"
        "<b>2) Nội dung chuyển khoản</b>\n"
        f"<code>{order_code} {user_id}</code>\n"
        "Ví dụ: <code>ORD-250820-12 123456789</code>\n\n"
        "<b>3) Xác nhận</b>\n"
        "• Sau khi chuyển khoản, bấm nút <b>“✅ Đã chuyển tiền”</b> bên dưới để báo admin.\n"
        "• Admin sẽ kích hoạt trong thời gian sớm nhất.\n\n"
        "<i>Lưu ý:</i>\n"
        "• Ghi đúng nội dung chuyển khoản để hệ thống so khớp nhanh.\n"
        "• Nếu sai nội dung, có thể cần bạn gửi ảnh biên lai khi admin yêu cầu.\n"
    )

def _send_paywall(bot: TeleBot, chat_id: int | str, user_id: int):
    order_code = _gen_order_code(user_id)
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton(
        "✅ Đã chuyển tiền",
        callback_data=f"PAID_CONFIRMED:{user_id}:{order_code}"
    ))
    bot.send_message(
        chat_id,
        _render_paywall_html(user_id, order_code),
        parse_mode='HTML',
        reply_markup=kb,
        disable_web_page_preview=True
    )

def _plus_link(bot: TeleBot) -> str:
    # DM-only: luôn trỏ về DM với deeplink UPGRADE (không dùng landing page)
    uname = bot.get_me().username
    return f"https://t.me/{uname}?start=UPGRADE"

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

        # 1) Deeplink UPGRADE -> tạo mã order + gửi Paywall + nút "Đã chuyển tiền"
        if token == "UPGRADE":
            _send_paywall(bot, m.chat.id, m.from_user.id)
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
                # Chưa Plus: gửi teaser + nút nâng cấp
                kb = types.InlineKeyboardMarkup()
                if m.chat.type == 'private':
                    kb.add(types.InlineKeyboardButton("✨ Nâng cấp/Gia hạn Plus", callback_data="OPEN_PAYWALL"))
                else:
                    kb.add(types.InlineKeyboardButton("✨ Nâng cấp/Gia hạn Plus", url=_plus_link(bot)))
                # ... send_message(..., reply_markup=kb)
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
        
    # /upgrade — mở Paywall trong DM
    @bot.message_handler(commands=['upgrade'])
    def on_upgrade(m):
        _send_paywall(bot, m.chat.id, m.from_user.id)

    # /plus_link — gửi link nâng cấp/gia hạn (DM)
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

    # Callback: "OPEN_PAYWALL" -> mở Paywall trong DM
    @bot.callback_query_handler(func=lambda c: c.data == "OPEN_PAYWALL")
    def on_open_paywall(call):
        _send_paywall(bot, call.message.chat.id, call.from_user.id)
        bot.answer_callback_query(call.id)

    # Callback: “Đã chuyển tiền”
    @bot.callback_query_handler(func=lambda c: c.data and c.data.startswith("PAID_CONFIRMED"))
    def on_paid_confirmed(call):
        # Hỗ trợ 2 hoặc 3 phần: PAID_CONFIRMED:<uid>[:<order_code>]
        try:
            parts = call.data.split(":")
            paid_uid = int(parts[1]) if len(parts) > 1 else call.from_user.id
            paid_order = parts[2] if len(parts) > 2 else "N/A"
        except Exception:
            paid_uid, paid_order = call.from_user.id, "N/A"

        bot.answer_callback_query(call.id, "Cảm ơn! Admin sẽ kích hoạt sớm.")
        # Báo cho admin
        u = call.from_user
        name = (u.first_name or "") + (" " + u.last_name if u.last_name else "")
        uname = f"@{u.username}" if u.username else "(no username)"
        text = (
            "💳 <b>Yêu cầu kích hoạt Plus</b>\n"
            f"• user_id: <code>{paid_uid}</code> {uname} {name}\n"
            f"• order_code: <code>{paid_order}</code>\n"
            f"• Lệnh gợi ý: <code>/plus_add {paid_uid} 30</code>\n"
        )

        for aid in _admin_ids():
            try:
                bot.send_message(aid, text, parse_mode='HTML')
            except Exception:
                pass
