from telebot import TeleBot, types

def mask(value):
    if isinstance(value, list):
        return "–".join(["X" for _ in value])
    return "X"

def post_teaser(bot: TeleBot, channel_id: int, signal: dict):
    text = (
        f"<b>{signal['symbol']} {signal['timeframe']}</b>\n"
        f"Setup: {signal['strategy']}\n"
        f"Entry: <spoiler>{mask(signal['entry'])}</spoiler> | "
        f"SL: <spoiler>{mask(signal['sl'])}</spoiler> | "
        f"TP: <spoiler>{mask(signal['tp'])}</spoiler>\n"
    )
    deep = f"https://t.me/{bot.get_me().username}?start=SIG_{signal['signal_id']}"
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("🔓 Xem đầy đủ", url=deep))
    msg = bot.send_message(chat_id=channel_id, text=text, parse_mode="HTML", reply_markup=kb)
    return msg.chat.id, msg.message_id
