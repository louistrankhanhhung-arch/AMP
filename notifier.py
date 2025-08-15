# notifier.py
from typing import NamedTuple, Optional
import time, itertools, os, requests

class PostRef(NamedTuple):
    chat_id: str | int
    message_id: int

class Notifier:
    def post(self, chat_id: str|int, text: str) -> PostRef: ...
    def reply(self, ref: PostRef, text: str) -> None: ...

# A. Console Notifier (dùng cho dev/test)
class ConsoleNotifier(Notifier):
    _seq = itertools.count(1)
    def post(self, chat_id, text):
        mid = next(self._seq)
        print(f"[telegram:post] chat={chat_id} mid={mid}\n{text}")
        return PostRef(chat_id, mid)
    def reply(self, ref, text):
        print(f"[telegram:reply] chat={ref.chat_id} reply_to={ref.message_id}\n{text}")

# B. Telegram Notifier (cắm sau)
class TelegramNotifier(Notifier):
    def __init__(self, token: Optional[str]=None, default_chat: Optional[str|int]=None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.default_chat = default_chat or os.getenv("TELEGRAM_CHAT_ID")
        if not self.token: raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
        if not self.default_chat: raise RuntimeError("Missing TELEGRAM_CHAT_ID")
        self.base = f"https://api.telegram.org/bot{self.token}"

    def post(self, chat_id=None, text=""):
        cid = chat_id or self.default_chat
        r = requests.post(f"{self.base}/sendMessage", data={"chat_id": cid, "text": text})
        r.raise_for_status()
        data = r.json()
        mid = data["result"]["message_id"]
        return PostRef(cid, mid)

    def reply(self, ref: PostRef, text=""):
        r = requests.post(f"{self.base}/sendMessage", data={
            "chat_id": ref.chat_id,
            "text": text,
            "reply_to_message_id": ref.message_id,
            "allow_sending_without_reply": True
        })
        r.raise_for_status()
