# main.py
# Quét – tạo – post tín hiệu lên Telegram; chỉ post khi action == ENTER


import os, threading, time, logging
import json
@@ -22,9 +23,16 @@
    from telegram_poster import Signal as TgSignal, DailyQuotaPolicy, post_signal
    from notifier import TelegramNotifier, PostRef
    from signal_tracker import SignalTracker

except Exception:
    TgSignal = DailyQuotaPolicy = post_signal = TelegramNotifier = PostRef = SignalTracker = None







# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

@@ -43,8 +51,15 @@
POLICY_DB = os.getenv("POLICY_DB", "/mnt/data/policy.sqlite3")
POLICY_KEY = os.getenv("POLICY_KEY", "global")

# Nút nâng cấp/FAQ thanh toán (tùy chọn)
JOIN_URL = os.getenv("JOIN_URL", None)








_BOT = None
_NOTIFIER = None
@@ -53,7 +68,7 @@
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID and TgSignal and DailyQuotaPolicy and post_signal:
    try:
        _NOTIFIER = TelegramNotifier(token=TELEGRAM_BOT_TOKEN, default_chat_id=TELEGRAM_CHANNEL_ID)
        _BOT = _NOTIFIER.bot
        _TRACKER = SignalTracker(_NOTIFIER)
        logging.info("[telegram] bot & tracker ready")
    except Exception as e:
@@ -108,174 +123,63 @@ def _build_structs_for(symbols: List[str]) -> List[Dict[str, Any]]:
    return out


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

                # Lấy strategy chuẩn (hạn chế 'GPT-plan')
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
                    join_btn_url=JOIN_URL,
                )

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
            out = make_telegram_signal(s4h, s1d, trigger_1h=s1h)
            print(json.dumps(out, ensure_ascii=False))
        return {"ok": True, "count": len(req.symbols)}
    scan_once_for_logs()
    return {"ok": True}


if __name__ == "__main__":
    scan_once_for_logs()
