# main.py
# Quét – tạo – post tín hiệu lên Telegram; chỉ post khi action == ENTER
# + DM flow: UPGRADE (Paywall), SIG_<id> unlock, /status, /plus_add (admin)

import os, threading, time, logging
import json
import traceback
from typing import List, Dict, Any, Iterable
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

# ====== modules trong repo ======
from kucoin_api import fetch_batch
from indicators import enrich_indicators, enrich_more
from structure_engine import build_struct_json
from universe import resolve_symbols
from gpt_signal_builder import make_telegram_signal

# (tuỳ chọn đăng Telegram & tracker)
try:
    from telegram_poster import Signal as TgSignal, DailyQuotaPolicy, post_signal
    from notifier import TelegramNotifier, PostRef
    from signal_tracker import SignalTracker
    from telebot import types
except Exception:
    TgSignal = DailyQuotaPolicy = post_signal = TelegramNotifier = PostRef = SignalTracker = None

# Membership
try:
    from membership import has_plus, activate_plus, get_expiry, remaining_days
except Exception:
    has_plus = activate_plus = get_expiry = remaining_days = None

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ====== FastAPI app ======
app = FastAPI()

# ====== Cấu hình ======
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "15"))
SCAN_TFS = os.getenv("SCAN_TFS", "1H,4H,1D")
MAX_GPT = int(os.getenv("MAX_GPT", "10"))
EXCHANGE = os.getenv("EXCHANGE", "KUCOIN")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID = int(os.getenv("TELEGRAM_CHANNEL_ID", "0") or 0)
POLICY_DB = os.getenv("POLICY_DB", "/mnt/data/policy.sqlite3")
POLICY_KEY = os.getenv("POLICY_KEY", "global")

# Nâng cấp không cần landing page: dùng DM /start UPGRADE
JOIN_URL = os.getenv("JOIN_URL", None)  # nếu không set, sẽ deep-link về DM
ADMIN_USER_IDS = set(int(x) for x in os.getenv("ADMIN_USER_IDS", "").split(",") if x.strip().isdigit())
ADMIN_NOTIFY_CHAT_ID = int(os.getenv("ADMIN_NOTIFY_CHAT_ID", "0") or 0)
PAYWALL_MSG = os.getenv(
    "PAYWALL_MSG",
    "Vui lòng chuyển khoản theo hướng dẫn hiển thị (ngân hàng, số tài khoản, nội dung).\n"
    "Sau khi chuyển, bấm nút <b>Đã chuyển tiền</b> để báo cho admin.\n"
)

_BOT = None
_NOTIFIER = None
_TRACKER = None

if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID and TgSignal and DailyQuotaPolicy and post_signal:
    try:
        _NOTIFIER = TelegramNotifier(token=TELEGRAM_BOT_TOKEN, default_chat_id=TELEGRAM_CHANNEL_ID)
        _BOT = _NOTIFIER.bot  # raw TeleBot instance
        _TRACKER = SignalTracker(_NOTIFIER)
        logging.info("[telegram] bot & tracker ready")
    except Exception as e:
        print("[telegram] init error:", e)
        _NOTIFIER = None
        _BOT = None
        _TRACKER = None


@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}


# ====== Utils ======
def _chunk(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


# --- Build structs cho 1H/4H/1D ---
from kucoin_api import fetch_ohlcv  # dùng riêng cho 1 symbol/TF


def _build_structs_for(symbols: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            df1h_raw = fetch_ohlcv(sym, timeframe="1H", limit=300)
            df4h_raw = fetch_ohlcv(sym, timeframe="4H", limit=300)
            df1d_raw = fetch_ohlcv(sym, timeframe="1D", limit=300)

            df1h = enrich_more(enrich_indicators(df1h_raw)) if df1h_raw is not None else None
            df4h = enrich_more(enrich_indicators(df4h_raw)) if df4h_raw is not None else None
            df1d = enrich_more(enrich_indicators(df1d_raw)) if df1d_raw is not None else None

            if df1h is None or df4h is None or df1d is None:
                print(f"[build] missing df for {sym}")
                continue
            if len(df1h) < 50 or len(df4h) < 50 or len(df1d) < 50:
                print(f"[build] not enough bars for {sym}")
                continue

            s1h = build_struct_json(sym, "1H", df1h)
            s4h = build_struct_json(sym, "4H", df4h)
            s1d = build_struct_json(sym, "1D", df1d)

            out.append({"symbol": sym, "1H": s1h, "4H": s4h, "1D": s1d})
        except Exception as e:
            print(f"[build] error {sym}:", e)
            traceback.print_exc()
    return out


# ====== Cache full payload để mở khóa qua DM ======
def _cache_signal_payload(tg_sig: "TgSignal"):
    try:
        idx_path = "/mnt/data/signals_index.json"
        idx = {}
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f) or {}
        idx[tg_sig.signal_id] = {
            "symbol": tg_sig.symbol,
            "side": tg_sig.side,
            "entries": tg_sig.entries or [],
            "sl": tg_sig.sl,
            "tps": tg_sig.tps or [],
            "leverage": tg_sig.leverage,
            "timeframe": tg_sig.timeframe,
            "eta": tg_sig.eta,
        }
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"[cache] save signal error: {e}")


def _load_signal_payload(signal_id: str):
    try:
        p = "/mnt/data/signals_index.json"
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            idx = json.load(f) or {}
        return idx.get(signal_id)
    except Exception:
        return None


def _render_full_from_payload(p: dict) -> str:
    def _fmt(v):
        if isinstance(v, (int, float)):
            s = f"{float(v):.8f}".rstrip("0").rstrip(".")
            return s or "0"
        return "-" if v is None else str(v)
    tps = "\n".join([f"<b>TP{i+1}:</b> {_fmt(v)}" for i, v in enumerate(p.get("tps") or [])])
    lev = f"\n<b>Leverage:</b> x{p['leverage']}" if p.get("leverage") else ""
    eta = f"\n<b>ETA:</b> {p['eta']}" if p.get("eta") else ""
    return (
        f"<b>#{p['symbol']}</b> — <b>{str(p.get('side','long')).upper()}</b> {p.get('timeframe','')}\n"
        f"<b>Entry:</b> {', '.join(_fmt(e) for e in (p.get('entries') or []))}\n"
        f"<b>SL:</b> {_fmt(p.get('sl'))}\n"
        f"{tps}{lev}{eta}"
    )


# ====== Scan & Post ======
def scan_once_for_logs():
    start_ts = datetime.utcnow().isoformat() + "Z"
    syms = resolve_symbols("")
    if not syms:
        pr
