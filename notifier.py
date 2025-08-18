# notifier.py
from typing import NamedTuple, Optional
import itertools
import os
import requests

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

# B. Telegram Notifier (sử dụng Bot API)
class TelegramNotifier(Notifier):
    def __init__(
        self,
        token: Optional[str] = None,
        default_chat: Optional[str | int] = None,
        # alias để tương thích code cũ/mới
        default_chat_id: Optional[str | int] = None,
        channel_id: Optional[str | int] = None,
        chat_id: Optional[str | int] = None,
        # tuỳ chọn khuyến nghị
        parse_mode: Optional[str] = None,
        disable_web_page_preview: Optional[bool] = None,
        timeout: Optional[float] = None,
    ):
        # Token
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")

        # Ưu tiên tham số truyền vào, sau đó tới ENV (hỗ trợ cả TELEGRAM_CHANNEL_ID & TELEGRAM_CHAT_ID)
        self.default_chat = (
            default_chat
            or default_chat_id
            or channel_id
            or chat_id
            or os.getenv("TELEGRAM_CHANNEL_ID")
            or os.getenv("TELEGRAM_CHAT_ID")
        )
        if not self.default_chat:
            raise RuntimeError("Missing TELEGRAM_CHANNEL_ID/TELEGRAM_CHAT_ID or default_chat")

        # Tuỳ chọn gửi
        self.parse_mode = parse_mode if parse_mode is not None else "HTML"
        self.disable_web_page_preview = True if disable_web_page_preview is None else bool(disable_web_page_preview)
        self.timeout = 10.0 if timeout is None else float(timeout)

        self.base = f"https://api.telegram.org/bot{self.token}"

    def _send(self, method: str, data: dict) -> dict:
        r = requests.post(f"{self.base}/{method}", data=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def post(self, chat_id=None, text: str = "") -> PostRef:
        cid = chat_id or self.default_chat
        payload = {
            "chat_id": cid,
            "text": text,
            "disable_web_page_preview": self.disable_web_page_preview,
        }
        if self.parse_mode:
            payload["parse_mode"] = self.parse_mode
        data = self._send("sendMessage", payload)
        mid = data["result"]["message_id"]
        return PostRef(cid, mid)

    def reply(self, ref: PostRef, text: str = "") -> None:
        payload = {
            "chat_id": ref.chat_id,
            "text": text,
            "reply_to_message_id": ref.message_id,
            "allow_sending_without_reply": True,
            "disable_web_page_preview": self.disable_web_page_preview,
        }
        if self.parse_mode:
            payload["parse_mode"] = self.parse_mode
        self._send("sendMessage", payload)
