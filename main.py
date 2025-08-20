# main.py
# Quét – tạo – post tín hiệu lên Telegram; chỉ post khi action == ENTER

import os
import threading
import time
import logging
import json
import traceback
import random
from typing import List, Dict, Any, Iterable
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel
from bot_handlers import register_handlers

# ====== modules trong repo ======
try:
    from indicators import enrich_indicators, enrich_more
except Exception:  # fallback an toàn nếu thiếu
    enrich_indicators = lambda x: x  # type: ignore
    enrich_more = lambda x: x  # type: ignore

try:
    from kucoin_api import fetch_ohlcv  # hàm lấy OHLCV theo TF
except Exception:
    fetch_ohlcv = None  # type: ignore

from universe import resolve_symbols
from gpt_signal_builder import make_telegram_signal

# (tuỳ chọn đăng Telegram & tracker)
try:
    from telegram_poster import Signal as TgSignal, DailyQuotaPolicy, post_signal
    from notifier import TelegramNotifier, PostRef
    from signal_tracker import SignalTracker
except Exception:
    TgSignal = DailyQuotaPolicy = post_signal = TelegramNotifier = PostRef = SignalTracker = None  # type: ignore

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ====== Config từ ENV ======
EXCHANGE = os.getenv("EXCHANGE", "kucoin")
SCAN_TFS = os.getenv("SCAN_TFS", "1H,4H,1D")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID = int(os.getenv("TELEGRAM_CHANNEL_ID", "0") or "0")

POLICY_DB = os.getenv("POLICY_DB", "/mnt/data/policy.sqlite3")
POLICY_KEY = os.getenv("POLICY_KEY", "global")

# Nút nâng cấp/FAQ thanh toán (tùy chọn)
JOIN_URL = os.getenv("JOIN_URL", None)

# Quét & giới hạn tải
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "20"))
MAX_GPT = int(os.getenv("MAX_GPT", "5"))
RATE_SLEEP_SECS = float(os.getenv("RATE_SLEEP_SECS", "1.2"))
GPT_MAX_RETRIES = int(os.getenv("GPT_MAX_RETRIES", "4"))

# Round-robin bền
RR_PTR_FILE = os.getenv("RR_PTR_FILE", "/mnt/data/rr_ptr.json")
RR_LOCK = threading.Lock()

# ====== App ======
_HANDLERS_WIRED = False
def wire_handlers_once(bot):
    global _HANDLERS_WIRED
    if bot and not _HANDLERS_WIRED:
        register_handlers(bot)
        _HANDLERS_WIRED = True
        
app = FastAPI()

_BOT: Any | None = None
_NOTIFIER: Any | None = None
_TRACKER: Any | None = None

if (
    TELEGRAM_BOT_TOKEN
    and TELEGRAM_CHANNEL_ID
    and TgSignal
    and DailyQuotaPolicy
    and post_signal
    and TelegramNotifier
    and PostRef
    and SignalTracker
):
    try:
        _NOTIFIER = TelegramNotifier(token=TELEGRAM_BOT_TOKEN, default_chat_id=TELEGRAM_CHANNEL_ID)  # type: ignore
        _BOT = _NOTIFIER.bot  # type: ignore
        wire_handlers_once(_BOT)
        _TRACKER = SignalTracker(_NOTIFIER)  # type: ignore
        logging.info("[telegram] bot & tracker ready")
    except Exception as e:
        logging.warning(f"[telegram] init failed: {e}")
        
# ====== Utils ======
def _chunk(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _load_rr_ptr(n_symbols: int) -> int:
    if n_symbols <= 0:
        return 0
    try:
        if os.path.exists(RR_PTR_FILE):
            with open(RR_PTR_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                return int(data.get("ptr", 0)) % n_symbols
    except Exception as e:
        logging.warning(f"[rr] load ptr error: {e}")
    return 0


def _save_rr_ptr(ptr: int) -> None:
    try:
        os.makedirs(os.path.dirname(RR_PTR_FILE), exist_ok=True)
        with open(RR_PTR_FILE, "w", encoding="utf-8") as f:
            json.dump({"ptr": int(ptr)}, f)
    except Exception as e:
        logging.warning(f"[rr] save ptr error: {e}")


def _pick_round_robin(symbols: List[str], k: int) -> List[str]:
    if not symbols or k <= 0:
        return []
    with RR_LOCK:
        n = len(symbols)
        ptr = _load_rr_ptr(n)
        order = symbols[ptr:] + symbols[:ptr]
        picked = order[: min(k, n)]
        _save_rr_ptr((ptr + len(picked)) % n)
    return picked


# --- Build structs cho 1H/4H/1D ---
def _build_structs_for(symbols: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            if fetch_ohlcv is None:  # type: ignore
                logging.warning("[build] fetch_ohlcv not available; skip %s", sym)
                continue

            # lấy dữ liệu từng khung
            df1h_raw = fetch_ohlcv(sym, timeframe="1H", limit=300)  # type: ignore
            df4h_raw = fetch_ohlcv(sym, timeframe="4H", limit=300)  # type: ignore
            df1d_raw = fetch_ohlcv(sym, timeframe="1D", limit=300)  # type: ignore

            # enrich cho mọi TF
            df1h = enrich_more(enrich_indicators(df1h_raw)) if df1h_raw is not None else None
            df4h = enrich_more(enrich_indicators(df4h_raw)) if df4h_raw is not None else None
            df1d = enrich_more(enrich_indicators(df1d_raw)) if df1d_raw is not None else None

            # thiếu (None) hoặc rỗng (0 rows) thì bỏ
            if any(d is None or getattr(d, "empty", False) for d in (df1h, df4h, df1d)):
                logging.info(f"[build] missing TF for {sym}")
                continue

            out.append({
                "symbol": sym,
                "1H": df1h,
                "4H": df4h,
                "1D": df1d,
            })
        except Exception as e:
            logging.warning(f"[build] error {sym}: {e}")
    return out


# ====== GPT call với exponential backoff ======
def _call_gpt_with_backoff(s4h, s1d, s1h):
    delay = RATE_SLEEP_SECS
    last_exc = None
    for _ in range(GPT_MAX_RETRIES + 1):
        try:
            return make_telegram_signal(s4h, s1d, trigger_1h=s1h)
        except Exception as e:
            last_exc = e
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                time.sleep(delay + random.random() * 0.3)
                delay = min(delay * 2, 15.0)
                continue
            raise
    raise last_exc if last_exc else RuntimeError("GPT call failed without exception")


# ====== Scan & Post ======
def scan_once_for_logs():
    start_ts = datetime.utcnow().isoformat() + "Z"
    syms = resolve_symbols("")
    if not syms:
        print("[scan] no symbols")
        return

    print(f"[scan] total symbols={len(syms)} exchange={EXCHANGE} tfs={SCAN_TFS}")

    structs = _build_structs_for(syms)
    picked = _pick_round_robin([x["symbol"] for x in structs], MAX_GPT)
    print(f"[scan] candidates(rr)={picked} (cap {MAX_GPT})")

    # Tạo policy 1 lần cho cả lượt quét
    policy = None
    if DailyQuotaPolicy:
        policy = DailyQuotaPolicy(db_path=POLICY_DB, key=POLICY_KEY)  # type: ignore

    sent = 0
    for sym in picked:
        try:
            s1h = next((x["1H"] for x in structs if x["symbol"] == sym), None)
            s4h = next((x["4H"] for x in structs if x["symbol"] == sym), None)
            s1d = next((x["1D"] for x in structs if x["symbol"] == sym), None)

            # thiếu TF thì bỏ qua, log rõ TF nào
            missing_tfs = [
                tf for tf, df in (("1H", s1h), ("4H", s4h), ("1D", s1d))
                if (df is None or (hasattr(df, "empty") and getattr(df, "empty")))
            ]
            if missing_tfs:
                print(f"[scan] missing structs: {sym} (missing {','.join(missing_tfs)})")
                continue

            # --- init để tránh UnboundLocalError nếu lỗi bên trong try ---
            out, tele, plan, meta, plans = {}, "", {}, {}, {}
            decision = {}
            
            try:
                out = _call_gpt_with_backoff(s4h, s1d, s1h)
                time.sleep(RATE_SLEEP_SECS)  # spacing hạ RPM
            
                tele = out.get("telegram_text")
                analysis_text = out.get("analysis_text") or ""
                
                # Lấy các nhánh có thể có trong output
                plan  = out.get("plan") or out.get("signal") or {}
                if not plan and isinstance(out.get("enter_plans"), list) and out["enter_plans"]:
                    plan = out["enter_plans"][0] or {}
                meta  = out.get("meta") or {}
                plans = out.get("plans") or {}
                p_intra = plans.get("intraday_1h") or plans.get("intraday") or {}
                p_swing = plans.get("swing_4h")    or plans.get("swing")    or {}
                
                # --- Fallback cho decision/action/side/conf ---
                decision = out.get("decision") or {}
                if not decision:
                    if plan:  # có plan => coi như ENTER
                        decision = {
                            "action": "ENTER",
                            "side": plan.get("side") or "long",
                            "confidence": (
                                plan.get("confidence")
                                or meta.get("intraday_conf")
                                or meta.get("swing_conf")
                            ),
                            "trigger": plan.get("trigger"),
                            "reason": plan.get("reason"),
                        }
                    else:
                        intr = str(meta.get("intraday_decision") or "").upper()
                        swg  = str(meta.get("swing_decision") or "").upper()
                        if "WAIT" in {intr, swg}:
                            guessed_action = "WAIT"
                        elif "AVOID" in {intr, swg}:
                            guessed_action = "AVOID"
                        else:
                            guessed_action = intr or swg or "WAIT"
                
                        side_guess = (
                            p_intra.get("side") or p_swing.get("side") or "none"
                        )
                        conf_guess = (
                            p_intra.get("confidence")
                            or p_swing.get("confidence")
                            or meta.get("intraday_conf")
                            or meta.get("swing_conf")
                        )
                        decision = {
                            "action": guessed_action,
                            "side": side_guess,
                            "confidence": conf_guess,
                        }
                
                # === Chuẩn hoá biến để log header ===
                action = str(decision.get("action") or "").upper()
                if not action:  # bảo đảm header không hiện N/A
                    intr = str(meta.get("intraday_decision") or "").upper()
                    swg  = str(meta.get("swing_decision") or "").upper()
                    action = "WAIT" if "WAIT" in {intr, swg} else ("AVOID" if "AVOID" in {intr, swg} else "N/A")
                
                side = (
                    str(decision.get("side") or "")
                    or plan.get("side")
                    or p_intra.get("side")
                    or p_swing.get("side")
                    or "none"
                )
                conf = (
                    decision.get("confidence")
                    or plan.get("confidence")
                    or meta.get("intraday_conf")
                    or meta.get("swing_conf")
                    or p_intra.get("confidence")
                    or p_swing.get("confidence")
                )
                
                # (tuỳ chọn) Lấy trigger/reason để dùng nếu cần
                trigger = (
                    decision.get("trigger")
                    or plan.get("trigger")
                    or p_intra.get("trigger_hint")
                    or p_swing.get("trigger_hint")
                    or meta.get("trigger_hint")
                    or meta.get("trigger")
                )
                reason = (
                    decision.get("reason")
                    or plan.get("reason")
                    or ((p_intra.get("reasons") or []) + (p_swing.get("reasons") or []) or [None])[0]
                )
                
                # === LOG HEADER + 2 BLOCK PHÂN TÍCH ===
                print(f"[signal] {sym} | {action} side={side} conf={conf}")
                if analysis_text:
                    print("[analysis]\n" + analysis_text)
                
                # === Gate: chỉ ENTER mới post Telegram ===
                if action != "ENTER":
                    continue

            
            except Exception as e:
                logging.exception(f"[scan] GPT err {sym}: {e}")
                # Giá trị mặc định khi lỗi GPT để không vỡ luồng
                decision = {"action": "AVOID", "side": "none", "confidence": 0.0}
            
                    
            # --- Chỉ ENTER mới tới đây: log setup gọn ---
            if tele:
                print("[signal]\n" + tele)
            else:
                entries = [ _fmt_num(p) for p in (plan.get("entries") or []) ]
                slv     = plan.get("sl")
                tps     = [ _fmt_num(p) for p in (plan.get("tps") or []) ]
                lev     = plan.get("leverage")
                eta     = plan.get("eta")
                fields = [
                    f"{sym}",
                    f"side={plan.get('side') or side}",
                    f"entry={','.join(entries)}" if entries else None,
                    f"sl={_fmt_num(slv)}" if slv is not None else None,
                    f"tp={','.join(tps)}" if tps else None,
                    f"lev=x{lev}" if lev else None,
                    f"eta={eta}" if eta else None,
                ]
                fields = [x for x in fields if x]
                print("[signal] " + " | ".join(fields))
            
            # (khối post Telegram giữ nguyên ở dưới)
            if _BOT and _NOTIFIER and TgSignal and post_signal:
                entries = plan.get("entries") or []
                slv = plan.get("sl")
                if not entries or slv is None:
                    print(f"[post_skip] {sym} missing entries/sl -> skip")
                    continue

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

                info = post_signal(  # type: ignore
                    bot=_BOT,  # type: ignore
                    channel_id=TELEGRAM_CHANNEL_ID,
                    sig=tg_sig,
                    policy=policy,  # type: ignore
                    join_btn_url=JOIN_URL,
                )
                
                # Đếm số tín hiệu đã post
                if info:
                    sent += 1
                
                # Ghi tracker (nếu có)
                if info and _TRACKER and PostRef:
                    post_ref = PostRef(chat_id=info["chat_id"], message_id=info["message_id"])  # type: ignore
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
                    _TRACKER.register_post(  # type: ignore
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
    logging.info(
        f"[scheduler] start: interval={SCAN_INTERVAL_MIN} min, MAX_GPT={MAX_GPT}, sleep={RATE_SLEEP_SECS}s, retries={GPT_MAX_RETRIES}"
    )
    while True:
        try:
            scan_once_for_logs()
            logging.info("[scan] done")
        except Exception as e:
            logging.exception(f"[scan] error: {e}")
        time.sleep(SCAN_INTERVAL_MIN * 60)


@app.on_event("startup")
def _on_startup():
    t = threading.Thread(target=_scan_loop, daemon=True)
    t.start()
    app.state.scan_thread = t
    logging.info("[scheduler] thread spawned")


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

            missing_tfs = [
                tf for tf, df in (("1H", s1h), ("4H", s4h), ("1D", s1d))
                if (df is None or (hasattr(df, "empty") and getattr(df, "empty")))
            ]
            if missing_tfs:
                print(f"[scan] missing structs: {sym} (missing {','.join(missing_tfs)})")
                continue

            out = _call_gpt_with_backoff(s4h, s1d, s1h)
            print(json.dumps(out, ensure_ascii=False))
        return {"ok": True, "count": len(req.symbols)}
    scan_once_for_logs()
    return {"ok": True}

if __name__ == "__main__":
    scan_once_for_logs()
