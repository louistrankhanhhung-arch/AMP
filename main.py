import os
import json
import time
import traceback
from typing import List, Dict, Any, Iterable
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

# ====== modules trong repo của bạn ======
from kucoin_api import fetch_batch
from indicators import enrich_indicators
from structure_engine import build_struct_json
from universe import resolve_symbols

from gpt_signal_builder import make_telegram_signal

# (tuỳ chọn đăng Telegram & nối dây tracker — giữ nguyên nếu đã dùng)
try:
    from telegram_poster import Signal as TgSignal, DailyQuotaPolicy, post_signal
    from notifier import TelegramNotifier, PostRef
    from signal_tracker import SignalTracker
except Exception:
    TgSignal = DailyQuotaPolicy = post_signal = TelegramNotifier = PostRef = SignalTracker = None


# ====== FastAPI app ======
app = FastAPI()


# ====== Cấu hình ======
SCAN_TFS = os.getenv("SCAN_TFS", "1H,4H,1D")  # đã đổi mặc định: có 1H
MAX_GPT = int(os.getenv("MAX_GPT", "10"))     # giới hạn số mã gửi GPT mỗi vòng để kiểm soát chi phí
EXCHANGE = os.getenv("EXCHANGE", "KUCOIN")

# Telegram (nếu muốn post)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID = int(os.getenv("TELEGRAM_CHANNEL_ID", "0") or 0)
POLICY_DB = os.getenv("POLICY_DB", "/mnt/data/policy.sqlite3")
POLICY_KEY = os.getenv("POLICY_KEY", "global")

_BOT = None
_NOTIFIER = None
_TRACKER = None

if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID and TgSignal and DailyQuotaPolicy and post_signal:
    try:
        _NOTIFIER = TelegramNotifier(token=TELEGRAM_BOT_TOKEN, default_chat_id=TELEGRAM_CHANNEL_ID)
        _BOT = _NOTIFIER.bot  # raw TeleBot instance
        _TRACKER = SignalTracker(_NOTIFIER)
    except Exception as e:
        print("[telegram] init error:", e)
        _NOTIFIER = None
        _BOT = None
        _TRACKER = None


# ====== Utils ======
def _chunk(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _build_structs_for(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Build struct JSON cho 3 khung 1H/4H/1D của mỗi symbol.
    Sử dụng indicators/enrich_indicators + structure_engine.build_struct_json (như bạn đã có).
    """
    # fetch theo batch để tiết kiệm call
    dfs_1h = fetch_batch(symbols, "1H")
    dfs_4h = fetch_batch(symbols, "4H")
    dfs_1d = fetch_batch(symbols, "1D")

    out: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            df1h = enrich_indicators(dfs_1h.get(sym))
            df4h = enrich_indicators(dfs_4h.get(sym))
            df1d = enrich_indicators(dfs_1d.get(sym))

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


def _pick_round_robin(symbols: List[str], k: int) -> List[str]:
    """
    Chọn k mã kiểu round-robin dựa trên epoch giờ hiện tại (đơn giản).
    """
    if not symbols:
        return []
    base = int(time.time() // 3600)  # thay đổi mỗi giờ
    start = base % len(symbols)
    order = symbols[start:] + symbols[:start]
    return order[:max(0, k)]


# ====== Scan ======
def scan_once_for_logs():
    """
    NO-FILTER, NO-TRIGGER:
    - Build struct 1H/4H/1D cho toàn bộ universe
    - Gửi GPT để xếp loại ENTER/WAIT/AVOID (và build setup khi ENTER)
    - ENTER: (tuỳ chọn) post Telegram + mở tracker
    - WAIT/AVOID: log
    """
    start_ts = datetime.utcnow().isoformat() + "Z"
    syms = resolve_symbols("")  # lấy danh sách 30 mã từ universe.py (ENV-first nếu có)
    if not syms:
        print("[scan] no symbols")
        return

    print(f"[scan] total symbols={len(syms)} exchange={EXCHANGE} tfs={SCAN_TFS}")

    # 1) Build structs
    structs = _build_structs_for(syms)

    # 2) Chọn danh sách gửi GPT theo round-robin (không filter cứng)
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

            # 3) Gọi GPT-4o: truyền 1H full struct (tham số trigger_1h giữ API cũ nhưng là struct 1H)
            out = make_telegram_signal(s4h, s1d, trigger_1h=s1h)

            if not out.get("ok"):
                print(f"[scan] GPT err {sym}: {out.get('error')}")
                continue

            # 4) Log kết quả & (tuỳ chọn) post Telegram
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
            print(f"[meta] {sym} conf={meta.get('confidence')} rr_min={rr.get('rr_min')} rr_max={rr.get('rr_max')} eta={meta.get('eta')}")

            # ==== Tuỳ chọn: Post Telegram & nối dây tracker ====
            if tele and _BOT and _NOTIFIER and TgSignal and DailyQuotaPolicy and post_signal:
                plan = out.get("plan") or out.get("signal") or {}
                tg_sig = TgSignal(
                    signal_id = plan.get("signal_id") or out.get("signal_id") or f"{sym.replace('/','')}-{int(time.time())}",
                    symbol    = sym.replace("/", ""),
                    timeframe = plan.get("timeframe") or "4H",
                    side      = plan.get("side") or side or "long",
                    strategy  = plan.get("strategy") or "GPT-plan",
                    entries   = plan.get("entries") or [],
                    sl        = plan.get("sl") if plan.get("sl") is not None else 0.0,
                    tps       = plan.get("tps") or [],
                    leverage  = plan.get("leverage"),
                    eta       = plan.get("eta"),
                )
                policy = DailyQuotaPolicy(db_path=POLICY_DB, key=POLICY_KEY)
                info = post_signal(bot=_BOT, channel_id=TELEGRAM_CHANNEL_ID, sig=tg_sig, policy=policy)
                if info and _TRACKER and PostRef:
                    # Sau khi post xong:
                    post_ref = PostRef(chat_id=info["chat_id"], message_id=info["message_id"])
                    
                    # Chuẩn hoá payload theo kỳ vọng của tracker:
                    signal_payload = {
                        "symbol": sym,                          # "AVAX/USDT"
                        "side": tg_sig.side,                    # "long"|"short"
                        "entries": tg_sig.entries or [],        # [ ... ]
                        "stop": tg_sig.sl,                      # float
                        "tps": tg_sig.tps or [],                # [ ... ]
                        "leverage": tg_sig.leverage,            # ví dụ "x5" (tuỳ bạn)
                    }
                    
                    # Map sl_mode: nếu plan trả "hard" thì tracker dùng "tick"
                    sl_mode = plan.get("sl_mode", "tick")
                    if sl_mode == "hard":
                        sl_mode = "tick"
                    
                    _TRACKER.register_post(
                        signal_id=tg_sig.signal_id,
                        ref=post_ref,
                        signal=signal_payload,
                        sl_mode=sl_mode,                        # "tick" | "close_4h"
                    )


        except Exception as e:
            print(f"[scan] error processing {sym}: {e}")
            traceback.print_exc()

    # 5) Lưu log gộp vòng quét
    try:
        os.makedirs("/mnt/data/gpt_logs", exist_ok=True)
        with open(f"/mnt/data/gpt_logs/scan_{int(time.time())}.meta.json", "w", encoding="utf-8") as f:
            json.dump({"at": start_ts, "picked": picked, "sent": sent}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[scan] write log error:", e)


# ====== API & CLI nho nhỏ ======
class ScanOnceReq(BaseModel):
    symbols: List[str] | None = None
    max_gpt: int | None = None


@app.post("/scan_once")
def api_scan_once(req: ScanOnceReq):
    global MAX_GPT
    if req.max_gpt:
        MAX_GPT = req.max_gpt
    if req.symbols:
        # Cho phép scan subset (debug)
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
    scan_once_for_logs()
