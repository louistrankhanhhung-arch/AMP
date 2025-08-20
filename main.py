# main.py
# Quét – tạo – post tín hiệu lên Telegram; chỉ post khi action == ENTER
# + DM flow: UPGRADE (Paywall), SIG_<id> unlock, /status, /plus_add (admin)
# + Khởi chạy scheduler bằng FastAPI lifespan (ổn định trên Railway)

import os, threading, time, logging
import json
import traceback
import contextlib
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
        print("[scan] no symbols")
        return

    print(f"[scan] total symbols={len(syms)} exchange={EXCHANGE} tfs={SCAN_TFS}")

    structs = _build_structs_for(syms)
    picked = [x["symbol"] for x in structs][:MAX_GPT]
    print(f"[scan] candidates(no-filter)={picked} (cap {MAX_GPT})")

    # Tạo policy 1 lần cho cả lượt quét
    policy = None
    if DailyQuotaPolicy:
        policy = DailyQuotaPolicy(db_path=POLICY_DB, key=POLICY_KEY)

    sent = 0
    for sym in picked:
        try:
            s1h = next((x["1H"] for x in structs if x["symbol"] == sym), None)
            s4h = next((x["4H"] for x in structs if x["symbol"] == sym), None)
            s1d = next((x["1D"] for x in structs if x["symbol"] == sym), None)
            if not (s1h and s4h and s1d):
                print(f"[scan] missing structs: {sym} (need 1H/4H/1D)")
                continue

            out = make_telegram_signal(s4h, s1d, trigger_1h=s1h)

            tele = out.get("telegram_text")
            decision = out.get("decision") or {}
            action = str(decision.get("action") or "").upper()
            side = str(decision.get("side") or "none")
            conf = decision.get("confidence")

            # Log ngắn
            if tele:
                print("[signal]\n" + tele)
            else:
                print(f"[signal] {sym} | {action or 'N/A'} side={side} conf={conf}")

            if out.get("analysis_text"):
                print("[analysis]\n" + out["analysis_text"])
            sent += 1

            # >>> Chỉ post khi action == ENTER <<<
            if action != "ENTER":
                print(f"[post_skip] {sym} action={action} -> skip")
                continue

            if _BOT and _NOTIFIER and TgSignal and post_signal and policy and out.get("ok"):
                plan = out.get("plan") or out.get("signal") or {}

                # Yêu cầu tối thiểu để post
                entries = plan.get("entries") or []
                slv = plan.get("sl")
                if not entries or slv is None:
                    print(f"[post_skip] {sym} missing entries/sl -> skip")
                    continue

                # Strategy: hạn chế 'GPT-plan'
                meta = out.get("meta") or {}
                strategy = (
                    plan.get("strategy")
                    or meta.get("strategy")
                    or meta.get("setup")
                    or decision.get("setup")
                    or "trend-follow"
                )

                tg_sig = TgSignal(
                    signal_id=plan.get("signal_id") or out.get("signal_id") or f"{sym.replace('/','')}-{int(time.time())}",
                    symbol=sym.replace("/", ""),
                    timeframe=plan.get("timeframe") or "4H",
                    side=plan.get("side") or side or "long",
                    strategy=strategy,
                    entries=entries,
                    sl=slv,
                    tps=plan.get("tps") or [],
                    leverage=plan.get("leverage"),
                    eta=plan.get("eta"),
                )

                info = post_signal(
                    bot=_BOT,
                    channel_id=TELEGRAM_CHANNEL_ID,
                    sig=tg_sig,
                    policy=policy,
                    join_btn_url=JOIN_URL,   # nếu None -> auto deep-link UPGRADE trong telegram_poster
                )

                # Cache để mở khóa qua DM (SIG_…)
                _cache_signal_payload(tg_sig)

                if info and _TRACKER and PostRef:
                    post_ref = PostRef(chat_id=info["chat_id"], message_id=info["message_id"])
                    signal_payload = {
                        "symbol": sym,
                        "side": tg_sig.side,
                        "entries": tg_sig.entries or [],
                        "stop": tg_sig.sl,
                        "tps": tg_sig.tps or [],
                        "leverage": tg_sig.leverage,
                    }
                    sl_mode = (plan.get("sl_mode") or "tick")
                    if sl_mode == "hard":
                        sl_mode = "tick"
                    _TRACKER.register_post(
                        signal_id=tg_sig.signal_id,
                        ref=post_ref,
                        signal=signal_payload,
                        sl_mode=sl_mode,
                    )

        except Exception as e:
            print(f"[scan] error processing {sym}: {e}")
            traceback.print_exc()

    try:
        os.makedirs("/mnt/data/gpt_logs", exist_ok=True)
        with open(f"/mnt/data/gpt_logs/scan_{int(time.time())}.meta.json", "w", encoding="utf-8") as f:
            json.dump({"at": start_ts, "picked": picked, "sent": sent}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[scan] write log error:", e)


# ====== DM FLOW ======
def _register_dm_handlers(bot):
    # /start SIG_<id> — mở khóa full nếu user có Plus
    @bot.message_handler(commands=["start"])
    def on_start(msg):
        parts = (msg.text or "").split(maxsplit=1)
        if len(parts) == 2 and parts[1].startswith("SIG_"):
            signal_id = parts[1][4:]
            if has_plus and has_plus(msg.from_user.id):
                payload = _load_signal_payload(signal_id)
                if payload:
                    bot.send_message(msg.chat.id, _render_full_from_payload(payload),
                                     parse_mode="HTML", disable_web_page_preview=True)
                else:
                    bot.send_message(msg.chat.id, "Không tìm thấy nội dung đầy đủ cho signal này. Thử lại sau.")
            else:
                kb = types.InlineKeyboardMarkup()
                kb.add(types.InlineKeyboardButton(
                    "✨ Nâng cấp/Gia hạn Plus",
                    url=f"https://t.me/{bot.get_me().username}?start=UPGRADE"))
                bot.send_message(
                    msg.chat.id,
                    "Nội dung này dành cho thành viên <b>Plus</b>. Bạn chưa có quyền xem.",
                    parse_mode="HTML",
                    reply_markup=kb
                )
            return

        if len(parts) == 2 and parts[1].upper() == "UPGRADE":
            kb = types.InlineKeyboardMarkup()
            kb.add(types.InlineKeyboardButton("✅ Đã chuyển tiền", callback_data=f"paid:{msg.from_user.id}"))
            bot.send_message(
                msg.chat.id,
                f"{PAYWALL_MSG}\n<b>Chat ID:</b> <code>{msg.from_user.id}</code>",
                parse_mode="HTML",
                reply_markup=kb,
                disable_web_page_preview=True
            )
            return

        bot.send_message(msg.chat.id, "Xin chào! Dùng /status để kiểm tra hạn Plus.")

    # User báo đã chuyển tiền -> gửi yêu cầu lên nhóm admin
    @bot.callback_query_handler(func=lambda c: c.data and c.data.startswith("paid:"))
    def on_paid(cb):
        try:
            uid = int(cb.data.split(":")[1])
        except Exception:
            bot.answer_callback_query(cb.id, "Lỗi dữ liệu.", show_alert=True)
            return

        bot.answer_callback_query(cb.id, "Đã ghi nhận. Admin sẽ xử lý sớm.")
        try:
            bot.edit_message_reply_markup(chat_id=cb.message.chat.id, message_id=cb.message.message_id, reply_markup=None)
        except Exception:
            pass
        bot.send_message(cb.message.chat.id, "Cảm ơn bạn! Admin sẽ kích hoạt Plus trong ít phút.")

        if ADMIN_NOTIFY_CHAT_ID:
            kb = types.InlineKeyboardMarkup()
            kb.add(types.InlineKeyboardButton("➕ Plus 7 ngày", callback_data=f"grant:{uid}:7"))
            kb.add(types.InlineKeyboardButton("➕ Plus 30 ngày", callback_data=f"grant:{uid}:30"))
            kb.add(types.InlineKeyboardButton("➕ Plus 90 ngày", callback_data=f"grant:{uid}:90"))
            txt = (
                "🔔 <b>YÊU CẦU NÂNG CẤP</b>\n"
                f"User: <a href=\"tg://user?id={uid}\">{uid}</a>\n"
                f"Chat ID: <code>{uid}</code>\n"
                f"Gợi ý lệnh: /plus_add {uid} 30"
            )
            bot.send_message(ADMIN_NOTIFY_CHAT_ID, txt, parse_mode="HTML", reply_markup=kb)

    # Admin bấm nút grant nhanh
    @bot.callback_query_handler(func=lambda c: c.data and c.data.startswith("grant:"))
    def on_grant(cb):
        if cb.from_user.id not in ADMIN_USER_IDS:
            bot.answer_callback_query(cb.id, "Bạn không có quyền.", show_alert=True)
            return
        try:
            _, uid_str, days_str = cb.data.split(":", 2)
            uid = int(uid_str); days = int(days_str)
            exp = activate_plus(uid, days=days) if activate_plus else None
        except Exception as e:
            bot.answer_callback_query(cb.id, f"Lỗi: {e}", show_alert=True)
            return

        bot.answer_callback_query(cb.id, f"Đã cấp Plus {days} ngày.", show_alert=True)
        bot.send_message(cb.message.chat.id, f"✅ Đã cấp Plus cho {uid} đến {exp.isoformat()} UTC.")
        try:
            bot.send_message(uid, f"Tài khoản đã được nâng cấp Plus {days} ngày. Hết hạn: {exp.isoformat()} UTC.")
        except Exception:
            pass

    @bot.message_handler(commands=["status"])
    def plus_status(msg):
        if not get_expiry:
            bot.send_message(msg.chat.id, "Membership chưa được bật trên server.")
            return
        exp = get_expiry(msg.from_user.id)
        if exp:
            days = remaining_days(msg.from_user.id)
            bot.send_message(msg.chat.id, f"Trạng thái: <b>Plus</b>\nHết hạn: {exp.isoformat()} UTC (~{days} ngày còn lại).",
                             parse_mode="HTML")
        else:
            bot.send_message(msg.chat.id, "Bạn chưa có Plus. Nhấn nút Nâng cấp/Gia hạn trong kênh để mở Paywall.", parse_mode="HTML")

    # Admin: /plus_add <chat_id> <days>
    @bot.message_handler(commands=["plus_add"])
    def plus_add(msg):
        if msg.from_user.id not in ADMIN_USER_IDS:
            return
        parts = (msg.text or "").split()
        if len(parts) != 3:
            bot.send_message(msg.chat.id, "Cú pháp: /plus_add <chat_id> <days>")
            return
        try:
            uid = int(parts[1]); days = int(parts[2])
            exp = activate_plus(uid, days=days) if activate_plus else None
            bot.send_message(msg.chat.id, f"OK. User {uid} Plus đến {exp.isoformat()} UTC.")
            try:
                bot.send_message(uid, f"Tài khoản đã được nâng cấp Plus {days} ngày. Hết hạn: {exp.isoformat()} UTC.")
            except Exception:
                pass
        except Exception as e:
            bot.send_message(msg.chat.id, f"Lỗi: {e}")


# ====== Scheduler background (lifespan) ======
def _scan_loop(stop_event: threading.Event):
    logging.info(f"[scheduler] start: interval={SCAN_INTERVAL_MIN} min")
    while not stop_event.is_set():
        try:
            scan_once_for_logs()
            logging.info("[scan] done")
        except Exception as e:
            logging.exception(f"[scan] error: {e}")
        stop_event.wait(SCAN_INTERVAL_MIN * 60)


stop_event = threading.Event()
scan_thread: threading.Thread | None = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global scan_thread
    # 1) spawn scheduler thread
    scan_thread = threading.Thread(target=_scan_loop, args=(stop_event,), daemon=True)
    scan_thread.start()
    logging.info("[scheduler] thread spawned")

    # 2) đăng ký DM handlers (nếu có bot)
    if _BOT:
        try:
            _register_dm_handlers(_BOT)
            logging.info("[dm] handlers registered")
        except Exception as e:
            logging.exception(f"[dm] register error: {e}")

    yield  # ----- app is running -----

    # shutdown
    try:
        stop_event.set()
        if scan_thread and scan_thread.is_alive():
            scan_thread.join(timeout=5)
        logging.info("[scheduler] thread stopped")
    except Exception:
        pass


# ====== FastAPI app với lifespan ======
app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}


# ====== API ======
class ScanOnceReq(BaseModel):
    symbols: List[str] | None = None
    max_gpt: int | None = None


@app.post("/scan_once")
def api_scan_once(req: ScanOnceReq):
    global MAX_GPT
    if req.max_gpt:
        MAX_GPT = req.max_gpt
    if req.symbols:
        structs = _build_structs_for(req.symbols)
        for sym in req.symbols:
            s1h = next((x["1H"] for x in structs if x["symbol"] == sym), None)
            s4h = next((x["4H"] for x in structs if x["symbol"] == sym), None)
            s1d = next((x["1D"] for x in structs if x["symbol"] == sym), None)
            out = make_telegram_signal(s4h, s1d, trigger_1h=s1h)
            print(json.dumps(out, ensure_ascii=False))
        return {"ok": True, "count": len(req.symbols)}
    scan_once_for_logs()
    return {"ok": True}


if __name__ == "__main__":
    # Chạy 1 vòng quét đơn lẻ khi chạy trực tiếp
    scan_once_for_logs()
