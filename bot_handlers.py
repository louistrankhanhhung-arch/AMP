from telebot import TeleBot
from app.membership import has_plus, activate_plus
from app.performance_logger import summarize_signal
from app.signal_tracker import render_full_signal_by_id

def register_handlers(bot: TeleBot):
    @bot.message_handler(commands=['start'])
    def on_start(m):
        token = None
        parts = m.text.split(maxsplit=1)
        if len(parts)>1: token = parts[1]
        if token and token.startswith('SIG_'):
            sig_id = token[4:]
            if has_plus(m.from_user.id):
                html = render_full_signal_by_id(sig_id)
                bot.send_message(m.chat.id, html, parse_mode='HTML')
            else:
                bot.send_message(m.chat.id, "Bạn đang ở gói Free. Gõ /plus_link để nâng cấp.")
        else:
            bot.send_message(m.chat.id, "Chào mừng! Gõ /status hoặc /plus_link để nâng cấp.")
