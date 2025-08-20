# main.py
# Quét – tạo – post tín hiệu lên Telegram; thêm DM flow membership & cache full signal

import os, threading, time, logging
import json
import traceback
from typing import List, Dict, Any, Iterable
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

# ====== modules trong repo ======
from kucoin_api import fetch_batch  # (có thể không dùng)
from indicators import enrich_indicators, enrich_more
from structure_engine import build_struct_json
from universe import resolve_symbols
from gpt_signal_builder import make_telegram_signal

# (tuỳ chọn đăng Telegram & nối dây tracker)
try:
    from telegram_poster import Signal as TgSignal, DailyQuotaPolicy, post_signal
    from notifier import TelegramNotifier, PostRef
    from signal_tracker import SignalTracker
except Exception:
    TgSignal = DailyQuotaPolicy = post_signal = TelegramNotifier = PostRef = SignalTracker = None

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

# Membership / DM
JOIN_URL = os.getenv("JOIN_URL", None)  # link nâng cấp/gia hạn Plus
ADMIN_USER_IDS = set(
    int(x) for x in os.getenv("ADMIN_USER_IDS", "").split(",") if x.strip().isdigit()
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
        yield lst[i : i + n]


# --- Build structs cho 1H/4H/1D ---
from kucoin_api import fetch_ohlcv  # dùng riêng cho 1 symbol/TF


def _build_structs_for(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Lấy OHLCV cho mỗi symbol ở 3 khung 1H/4H/1D.
    Bổ sung enrich_more(...) để có thêm candle anatomy, vol_z20, soft MA...
    """
    out: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            # lấy dữ liệu từng khung
            df1h_raw = fetch_ohlcv(sym, timeframe="1H", limit=300)
            df4h_raw = fetch_ohlcv(sym, timeframe="4H", limit=300)
            df1d_raw = fetch_ohlcv(sym, timeframe="1D", limit=300)

            # enrich cho mọi TF
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


# ====== Round-Robin bền ======
RR_PTR_FILE = os.getenv("RR_PTR_FILE", "/mnt/data/rr_ptr.json")
RR_LOCK = threading.Lock()


def _load_rr_ptr(n_symbols: int) -> int:
    """Đọc con trỏ round-robin từ file; trả 0 nếu chưa có."""
    if n_symbols <= 0:
        return 0
    try:
        if os.path.exists(RR_PTR_FILE):
            with open(RR_PTR_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                ptr = int(data.get("ptr", 0))
                return max(0, ptr) % n_symbols
    except Exception as e:
        logging.warning(f"[rr] load ptr error: {e}")
    return 0


def _save_rr_ptr(ptr: int) -> None:
    """Ghi con trỏ round-robin ra file (best-effort)."""
    try:
        os.makedirs(os.path.dirname(RR_PTR_FILE), exist_ok=True)
        with open(RR_PTR_FILE, "w", encoding="utf-8") as f:
            json.dump({"ptr": int(ptr)}, f)
    except Exception as e:
        logging.warning(f"[rr] save ptr error: {e}")


def _pick_round_robin(symbols: List[str], k: int) -> List[str]:
    """Chọn k mã theo con trỏ lưu file; lần sau tiếp tục từ vị trí mới (bền qua restart)."""
    if not symbols or k <= 0:
        return []
    with RR_LOCK:
        n = len(symbols)
        ptr = _load_rr_ptr(n)
        order = symbols[ptr:] + symbols[:ptr]
        picked = order[: min(k, n)]
        new_ptr = (ptr + len(picked)) % n
        _save_rr_ptr(new_ptr)
    return picked


# ====== Scan & Post ======
def _cache_signal_payload(tg_sig: "TgSignal"):
    """
    Lưu payload đầy đủ vào /mnt/data/signals_index.json để DM có thể mở khóa theo signal_id.
    """
    try:
        idx_path = "/mnt/data/signals_index.json"
        idx: Dict[str, Any] = {}
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


def scan_once_for_logs():
    start_ts = datetime.utcnow().isoformat() + "Z"
    syms = resolve_symbols("")
    if not syms:
        print("[scan] no symbols")
        return

    print(f"[scan] total symbols={len(syms)} exchange={EXCHANGE} tfs={SCAN_TFS}")

    structs = _build_structs_for(syms)
    picked = _pick_round_robin([x["symbol"] for x in structs], MAX_GPT)
    print(f"[scan] candidates(no-filter)={picked} (cap {MAX_GPT})")

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

            if not out.get("ok"):
                print(f"[scan] GPT err {sym}: {out.get('error')}")
                continue

            tele = out.get("telegram_text")
            decision = out.get("decision") or {}
            act = str(decision.get("action") or "N/A").upper()
            side = str(decision.get("side") or "none")
            conf = decision.get("confidence")

            if tele:
                print("[signal]\n" + tele)
            else:
                print(f"[signal] {sym} | {act} side={side} conf={conf}")

            if out.get("analysis_text"):
                print("[analysis]\n" + out["analysis_text"])
            sent += 1

            meta = out.get("meta", {})
            rr = meta.get("rr", {})
            print(
                f"[meta] {sym} conf={meta.get('confidence')} rr_min={rr.get('rr_min')} "
                f"rr_max={rr.get('rr_max')} eta={meta.get('eta')}"
            )

            if tele and _BOT and _NOTIFIER and TgSignal and DailyQuotaPolicy and post_signal:
                plan = out.get("plan") or out.get("signal") or {}
                tg_sig = TgSignal(
                    signal_id=plan.get("signal_id") or out.get("signal_id") or f"{sym.replace('/','')}-{int(time.time())}",
                    symbol=sym.replace("/", ""),
                    timeframe=plan.get("timeframe") or "4H",
                    side=plan.get("side") or side or "long",
                    strategy=plan.get("strategy") or "GPT-plan",
                    entries=plan.get("entries") or [],
                    sl=plan.get("sl") if plan.get("sl") is not None else 0.0,
                    tps=plan.get("tps") or [],
                    leverage=plan.get("leverage"),
                    eta=plan.get("eta"),
                )
                policy = DailyQuotaPolicy(db_path=POLICY_DB, key=POLICY_KEY)

                # >>> THAY ĐỔI: truyền join_btn_url để hiện nút nâng cấp/gia hạn Plus <<<
                info = post_signal(
                    bot=_BOT,
                    channel_id=TELEGRAM_CHANNEL_ID,
                    sig=tg_sig,
                    policy=policy,
                    join_btn_url=JOIN_URL,
                )

                # Lưu cache full nội dung theo signal_id để DM có thể mở khóa
                _cache_signal_payload(tg_sig)

                # Nối tracker nếu có
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


# ====== Scheduler background ======
def _scan_loop():
    logging.info(f"[scheduler] start: interval={SCAN_INTERVAL_MIN} min")
    while True:
        try:
            scan_once_for_logs()
            logging.info("[scan] done")
        except Exception as e:
            logging.exception(f"[scan] error: {e}")
        time.sleep(SCAN_INTERVAL_MIN * 60)  # phút → giây


@app.on_event("startup")
def _on_startup():
    # Spawn thread quét định kỳ
    t = threading.Thread(target=_scan_loop, daemon=True)
    t.start()
    app.state.scan_thread = t
    logging.info("[scheduler] thread spawned")

    # Đăng ký DM handlers — chỉ khi bot đã sẵn sàng
    if _BOT:
        _register_dm_handlers(_BOT)


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


# ====== DM FLOW: deep-link SIG_, /plus, /status, /redeem, /grant_plus ======
def _register_dm_handlers(bot):
    from telebot import types  # import nội bộ để tránh lỗi khi không bật Telegram
    from membership import has_plus, activate_plus, get_expiry, remaining_days

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

    @bot.message_handler(commands=["start"])
    def on_start(msg):
        # Deep-link: /start SIG_<id>
        parts = (msg.text or "").split(maxsplit=1)
        if len(parts) == 2 and parts[1].startswith("SIG_"):
            signal_id = parts[1][4:]
            if has_plus(msg.from_user.id):
                payload = _load_signal_payload(signal_id)
                if payload:
                    bot.send_message(
                        msg.chat.id,
                        _render_full_from_payload(payload),
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                else:
                    bot.send_message(msg.chat.id, "Không tìm thấy nội dung đầy đủ cho signal này. Thử lại sau.")
            else:
                kb = types.InlineKeyboardMarkup()
                if JOIN_URL:
                    kb.add(types.InlineKeyboardButton("✨ Nâng cấp/Gia hạn Plus", url=JOIN_URL))
                bot.send_message(
                    msg.chat.id,
                    "Nội dung này dành cho thành viên <b>Plus</b>. "
                    "Bạn chưa có quyền xem. Nâng cấp để mở khóa toàn bộ Entry/SL/TP.",
                    parse_mode="HTML",
                    reply_markup=kb,
                )
        else:
            bot.send_message(msg.chat.id, "Xin chào! Gõ /plus để xem gói, /status để kiểm tra hạn.")

    @bot.message_handler(commands=["plus"])
    def plus_info(msg):
        text = (
            "<b>Plus Membership</b>\n"
            "• Xem full tất cả tín hiệu (Entry/SL/TP)\n"
            "• Không bị che số, có ETA & ghi chú\n"
            "• Hỗ trợ ưu tiên\n\n"
            "Gõ /status để xem hạn."
        )
        if JOIN_URL:
            kb = types.InlineKeyboardMarkup()
            kb.add(types.InlineKeyboardButton("✨ Nâng cấp/Gia hạn Plus", url=JOIN_URL))
            bot.send_message(msg.chat.id, text, parse_mode="HTML", reply_markup=kb)
        else:
            bot.send_message(msg.chat.id, text, parse_mode="HTML")

    @bot.message_handler(commands=["status"])
    def plus_status(msg):
        exp = get_expiry(msg.from_user.id)
        if exp:
            days = remaining_days(msg.from_user.id)
            bot.send_message(
                msg.chat.id,
                f"Trạng thái: <b>Plus</b>\nHết hạn: {exp.isoformat()} UTC (~{days} ngày còn lại).",
                parse_mode="HTML",
            )
        else:
            bot.send_message(msg.chat.id, "Bạn chưa có Plus. Gõ /plus để xem quyền lợi.", parse_mode="HTML")

    @bot.message_handler(commands=["redeem"])
    def redeem_code(msg):
        # Dùng ENV PLUS_VOUCHERS='{"CODE30":30,"TRIAL7":7}'
        try:
            parts = (msg.text or "").split(maxsplit=1)
            if len(parts) < 2:
                bot.send_message(msg.chat.id, "Cú pháp: /redeem MÃ_VOUCHER")
                return
            codes = json.loads(os.getenv("PLUS_VOUCHERS", "{}") or "{}")
            code = parts[1].strip().upper()
            days = int(codes.get(code, 0))
            if days <= 0:
                bot.send_message(msg.chat.id, "Mã không hợp lệ.")
                return
            exp = activate_plus(msg.from_user.id, days=days)
            bot.send_message(msg.chat.id, f"Đã kích hoạt Plus thêm {days} ngày. Hết hạn: {exp.isoformat()} UTC.")
        except Exception as e:
            bot.send_message(msg.chat.id, f"Lỗi redeem: {e}")

    @bot.message_handler(commands=["grant_plus"])
    def grant_plus(msg):
        # Admin cấp tay: /grant_plus <user_id> <days>
        try:
            if msg.from_user.id not in ADMIN_USER_IDS:
                return
            parts = (msg.text or "").split()
            if len(parts) != 3:
                bot.send_message(msg.chat.id, "Cú pháp: /grant_plus <user_id> <days>")
                return
            uid = int(parts[1])
            days = int(parts[2])
            exp = activate_plus(uid, days=days)
            bot.send_message(msg.chat.id, f"OK. User {uid} Plus đến {exp.isoformat()} UTC.")
        except Exception as e:
            bot.send_message(msg.chat.id, f"Lỗi grant_plus: {e}")


if __name__ == "__main__":
    # Chạy 1 vòng quét đơn lẻ khi chạy trực tiếp
    scan_once_for_logs()
