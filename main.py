import argparse
import json
import os
import time
import threading
import traceback
from datetime import datetime
from typing import List, Dict

# --- your modules ---
from kucoin_api import fetch_batch
from indicators import enrich_indicators, enrich_more, calc_vp, fetch_funding_oi
from structure_engine import build_struct_json
from filter import rank_all
from universe import resolve_symbols  # dùng hàm chuẩn hoá danh mục mã
from gpt_signal_builder import make_telegram_signal
from telebot import TeleBot
from notifier import TelegramNotifier, PostRef
from telegram_poster import Signal as TgSignal, DailyQuotaPolicy, post_signal
from signal_tracker import SignalTracker
from price_watcher import PriceWatcher
from glue_tick import wire_symbol

# ---------------------------
# helpers
# ---------------------------
def _tflist(s: str) -> List[str]:
    return [x.strip().upper() for x in (s or "").split(",") if x.strip()]


def build_structs_for_symbol(
    symbol: str,
    tfs: List[str],
    limit: int = 300,
    with_futures: bool = False,
    with_liquidity: bool = False,
) -> List[Dict]:
    """
    Build structs cho 1 symbol.
    Nếu có 1D trong tfs, dùng làm context cho 4H.
    """
    batch = fetch_batch(symbol, timeframes=tfs, limit=limit)

    # context 1D (nếu có) cho 4H
    ctx_df = batch.get("1D")
    if ctx_df is not None:
        ctx_df = enrich_more(enrich_indicators(ctx_df))

    futures_sent = fetch_funding_oi(symbol) if with_futures else None

    out: List[Dict] = []
    for tf in tfs:
        df = batch.get(tf)
        if df is None:
            continue
        df = enrich_more(enrich_indicators(df))
        lz = calc_vp(df) if with_liquidity else None
        context_df = ctx_df if (tf == "4H" and ctx_df is not None) else None

        struct = build_struct_json(
            symbol, tf, df,
            context_df=context_df,
            liquidity_zones=lz,
            futures_sentiment=futures_sent
        )
        out.append(struct)
    return out


# ---------------------------
# CLI mode (chạy local nhanh)
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['onboard', 'coverage'], default='onboard')
    ap.add_argument('--symbol', default='BTC/USDT')
    ap.add_argument('--tfs', default='4H,1D', help='Comma-separated, e.g. 4H,1D')
    ap.add_argument('--limit', type=int, default=100, help='Candles per timeframe')
    ap.add_argument('--with-futures', action='store_true', help='Attach funding/OI from KuCoin Futures')
    ap.add_argument('--with-liquidity', action='store_true', help='Attach liquidity zones (volume profile)')
    args = ap.parse_args()

    tfs = _tflist(args.tfs)
    structs = build_structs_for_symbol(
        args.symbol, tfs, limit=args.limit,
        with_futures=args.with_futures,
        with_liquidity=args.with_liquidity
    )
    # Xếp hạng & in kết quả
    ranks = rank_all(structs)
    print(json.dumps([r.__dict__ for r in ranks], ensure_ascii=False, indent=2))


# ---------------------------
# FastAPI app (cho Railway)
# ---------------------------
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": datetime.utcnow().isoformat() + "Z"}


@app.get("/structs.json")
def structs(
    symbols: str = "",
    tfs: str = "4H,1D",
    limit: int = 300,
    with_futures: int = 0,
    with_liquidity: int = 0
):
    syms = resolve_symbols(symbols)                  # <= chỉ gọi resolve_symbols
    tflist = _tflist(tfs)
    out = []
    for sym in syms:
        out.extend(build_structs_for_symbol(
            sym, tflist, limit=limit,
            with_futures=bool(with_futures),
            with_liquidity=bool(with_liquidity)
        ))
    return {"generated_at": datetime.utcnow().isoformat()+"Z", "structs": out}


@app.get("/ranks.json")
def ranks(
    symbols: str = "",
    tfs: str = "4H,1D",
    limit: int = 300,
    with_futures: int = 0,
    with_liquidity: int = 0
):
    syms = resolve_symbols(symbols)                  # <= chỉ gọi resolve_symbols
    tflist = _tflist(tfs)
    all_structs = []
    for sym in syms:
        all_structs.extend(build_structs_for_symbol(
            sym, tflist, limit=limit,
            with_futures=bool(with_futures),
            with_liquidity=bool(with_liquidity)
        ))
    rks = rank_all(all_structs)
    return {"generated_at": datetime.utcnow().isoformat()+"Z",
            "ranks":[r.__dict__ for r in rks]}


@app.get("/bucketA_structs.json")
def bucketA_structs(
    symbols: str = "",
    tfs: str = "4H,1D",
    limit: int = 300,
    with_futures: int = 0,
    with_liquidity: int = 0,
    min_bucket: str = "A",
    min_score: float = 7.0
):
    syms = resolve_symbols(symbols)                  # <= chỉ gọi resolve_symbols
    tflist = _tflist(tfs)

    all_structs = []
    for sym in syms:
        all_structs.extend(build_structs_for_symbol(
            sym, tflist, limit=limit,
            with_futures=bool(with_futures),
            with_liquidity=bool(with_liquidity)
        ))

    rks = rank_all(all_structs)

    def _ord(b): return {"A":3,"B":2,"C":1}.get((b or "C").upper(), 0)
    ok_syms = set()
    for r in rks:
        bucket = getattr(r, "bucket_best", None) or getattr(r, "bucket", None)
        score  = getattr(r, "score_best", None)  or getattr(r, "score", None)
        if (bucket and _ord(bucket) >= _ord(min_bucket)) or (isinstance(score,(int,float)) and score >= float(min_score)):
            ok_syms.add(r.symbol)

    filtered = [s for s in all_structs if s.get("symbol") in ok_syms]
    return {
        "generated_at": datetime.utcnow().isoformat()+"Z",
        "symbols": sorted(ok_syms),
        "structs": filtered,
        "ranks": [r.__dict__ for r in rks if r.symbol in ok_syms]
    }

@app.get("/gpt_signal.json")
def gpt_signal(symbol: str, tfs: str = "4H,1D", limit: int = 300, with_futures: int = 0, with_liquidity: int = 0):
    tflist = [x.strip().upper() for x in tfs.split(",") if x.strip()]
    if "4H" not in tflist or "1D" not in tflist:
        raise HTTPException(status_code=400, detail="tfs phải gồm 4H và 1D")
    structs = build_structs_for_symbol(symbol, tflist, limit=limit, with_futures=bool(with_futures), with_liquidity=bool(with_liquidity))
    s4 = next((s for s in structs if s.get("timeframe")=="4H"), None)
    s1 = next((s for s in structs if s.get("timeframe")=="1D"), None)
    if not s4 or not s1:
        return {"ok": False, "error": "missing 4H or 1D struct"}

    # trigger 1H (nếu cần)
    trigger_1h = None
    if USE_1H_TRIGGER:
        trigger_1h = _build_trigger_1h(symbol)

    out = make_telegram_signal(s4, s1, trigger_1h=trigger_1h)
    return out


# ====== Background scanner writing signals to logs ======

# Cấu hình lịch
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "60"))
SCAN_TFS          = os.getenv("SCAN_TFS", "4H,1D")
SCAN_LIMIT        = int(os.getenv("SCAN_LIMIT", "200"))
MIN_BUCKET        = os.getenv("MIN_BUCKET", "A")
MIN_SCORE         = float(os.getenv("MIN_SCORE", "7"))
MAX_GPT           = int(os.getenv("MAX_GPT", "8"))

# 1H trigger config
USE_1H_TRIGGER    = int(os.getenv("USE_1H_TRIGGER", "1"))     # 1 = bật 1H cho các mã đậu filter
TRIGGER_1H_LIMIT  = int(os.getenv("TRIGGER_1H_LIMIT", "180"))

def _bucket_ord(b: str) -> int:
    return {"A":3, "B":2, "C":1}.get((b or "C").upper(), 0)

# ---------- Telegram wiring (khởi tạo 1 lần, dùng cho toàn app) ----------
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID  = int(os.getenv("TELEGRAM_CHANNEL_ID", "0"))
JOIN_URL             = os.getenv("JOIN_URL", None)
POLICY_DB            = os.getenv("POLICY_DB", "/mnt/data/policy.sqlite3")
POLICY_KEY           = os.getenv("POLICY_KEY", "global")

_BOT      = TeleBot(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID else None
_NOTIFIER = TelegramNotifier() if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID else None
_POLICY   = DailyQuotaPolicy(db_path=POLICY_DB, key=POLICY_KEY)
_WATCHER  = PriceWatcher()


# ---- round-robin helper ----
ROUND_ROBIN = {"i": 0}
def _pick_round_robin(cands: List[str], cap: int) -> List[str]:
    n = len(cands)
    if n == 0 or cap <= 0:
        return []
    i = ROUND_ROBIN["i"] % n
    if cap >= n:
        ROUND_ROBIN["i"] = 0
        return cands[:]
    if i + cap <= n:
        out = cands[i:i+cap]
    else:
        out = cands[i:] + cands[:(i+cap) % n]
    ROUND_ROBIN["i"] = (i + cap) % n
    return out

# ---- trigger 1H helper ----
def _build_trigger_1h(symbol: str) -> Dict | None:
    """
    Ưu tiên dùng trigger_1H.py nếu bạn đã có.
    Nếu không, fallback: fetch 1H, enrich, tạo vài flag đơn giản cho GPT.
    """
    try:
        from trigger_1H import build_trigger_for_symbol  # optional
        return build_trigger_for_symbol(symbol, limit=TRIGGER_1H_LIMIT)
    except Exception:
        pass
    # fallback
    try:
        batch = fetch_batch(symbol, timeframes=["1H"], limit=TRIGGER_1H_LIMIT)
        df = batch.get("1H")
        if df is None or len(df) == 0:
            return None
        df = enrich_more(enrich_indicators(df))
        last = df.iloc[-1]
        trig = {
            "symbol": symbol,
            "timeframe": "1H",
            "reclaim_ma20": bool(last["close"] > last["ema20"]),
            "rsi_gt_50": bool((last.get("rsi", 0) or 0) > 50),
            "note": "fallback-1H",
        }
        return trig
    except Exception as e:
        print(f"[scan] trigger_1h fallback error {symbol}: {e}")
        return None


def scan_once_for_logs():
    """Quét toàn bộ SYMBOLS → lọc bucket/score → gọi GPT → IN RA LOGS."""
    start_ts = datetime.utcnow().isoformat() + "Z"
    syms = resolve_symbols("")  # ENV-first
    tflist = _tflist(SCAN_TFS)

    print(f"[scan] start ts={start_ts} symbols={len(syms)} tfs={tflist} limit={SCAN_LIMIT} "
          f"min_bucket={MIN_BUCKET} min_score={MIN_SCORE}")

    # 1) Build structs (4H/1D) cho tất cả
    all_structs = []
    for s in syms:
        try:
            all_structs.extend(build_structs_for_symbol(
                s, tflist, limit=SCAN_LIMIT,
                with_futures=False, with_liquidity=False
            ))
        except Exception as e:
            print(f"[scan] build_structs error symbol={s}: {e}")

    # 2) Rank & lọc theo bucket/score
    rks = rank_all(all_structs)
    ok_syms = set()
    for r in rks:
        bucket = getattr(r, "bucket_best", None) or getattr(r, "bucket", None)
        score  = getattr(r, "score_best", None)  or getattr(r, "score", None)
        if (bucket and _bucket_ord(bucket) >= _bucket_ord(MIN_BUCKET)) or (
            isinstance(score, (int,float)) and score >= float(MIN_SCORE)
        ):
            ok_syms.add(r.symbol)

    if not ok_syms:
        print("[scan] no candidates passing filters")
        return {"ok": True, "count": 0}

    # 3) Chọn MAX_GPT theo round-robin
    cands = [s for s in syms if s in ok_syms]
    picked = _pick_round_robin(cands, MAX_GPT)
    print(f"[scan] candidates={list(picked)} (cap {MAX_GPT})")

    sent = 0
    for sym in picked:
        try:
            s4 = next((x for x in all_structs if x.get("symbol")==sym and x.get("timeframe")=="4H"), None)
            s1 = next((x for x in all_structs if x.get("symbol")==sym and x.get("timeframe")=="1D"), None)
            if not s4 or not s1:
                print(f"[scan] missing structs: {sym}")
                continue

            trigger_1h = _build_trigger_1h(sym) if USE_1H_TRIGGER else None

            out = make_telegram_signal(s4, s1, trigger_1h=trigger_1h)
            if not out.get("ok"):
                print(f"[scan] GPT err {sym}: {out.get('error')}")
                continue

            tele     = out.get("telegram_text")
            decision = out.get("decision") or {}   # đảm bảo là dict

            # --- ĐĂNG TELEGRAM + NỐI DÂY TRACKER ---
            if tele and _BOT and _NOTIFIER:
                # Ưu tiên dùng plan/signal chi tiết nếu có để post dạng teaser/full qua telegram_poster
                plan = out.get("plan") or out.get("signal") or {}
                try:
                    tg_sig = TgSignal(
                        signal_id = plan.get("signal_id") or out.get("signal_id") or f"{sym.replace('/','')}-{int(time.time())}",
                        symbol    = sym.replace("/", ""),
                        timeframe = plan.get("timeframe") or "1H",
                        side      = plan.get("side") or (decision.get("side") or "long"),
                        strategy  = plan.get("strategy") or "Auto-plan",
                        entries   = plan.get("entries") or [],
                        sl        = plan.get("sl") if plan.get("sl") is not None else 0.0,
                        tps       = plan.get("tps") or [],
                        leverage  = plan.get("leverage"),
                        eta       = plan.get("eta"),
                    )
                    # Đăng bài lên channel để lấy chat_id/message_id
                    info = post_signal(
                        bot=_BOT,
                        channel_id=TELEGRAM_CHANNEL_ID,
                        sig=tg_sig,
                        policy=_POLICY,
                        max_free_per_day=2,
                        min_plus_between_free=5,
                        force_free=False,
                        join_btn_url=JOIN_URL,
                    )
                    post_ref = PostRef(info["chat_id"], info["message_id"])

                    # Mở tracker & nối watcher
                    tracker = SignalTracker(_NOTIFIER)
                    tracker.open_signal(
                        signal_id=tg_sig.signal_id,
                        symbol=sym,                         # giữ "SUI/USDT" cho watcher
                        side=tg_sig.side,
                        entries=tg_sig.entries,
                        stop=tg_sig.sl,
                        tps=tg_sig.tps,
                        post_ref=post_ref,
                        sl_mode=plan.get("sl_mode", "hard"),
                    )
                    wire_symbol(_WATCHER, tracker, sym)
                    sent += 1
                    # In thêm bản text để dễ debug
                    print("[signal]\n" + tele)
                    if out.get("analysis_text"):
                        print("[analysis]\n" + out["analysis_text"])
                except Exception as ee:
                    # Nếu thiếu dữ liệu để post theo poster, fallback in log như cũ
                    print(f"[signal] post/wire fallback for {sym}: {ee}")
                    print("[signal]\n" + tele)
                    if out.get("analysis_text"):
                        print("[analysis]\n" + out["analysis_text"])
            else:
                # Không có telegram_text hoặc chưa cấu hình bot/channel → giữ nguyên log cũ
                if tele:
                    print("[signal]\n" + tele)
                    if out.get("analysis_text"):
                        print("[analysis]\n" + out["analysis_text"])
                else:
                    act  = str(decision.get("action") or "N/A").upper()
                    side = str(decision.get("side") or "none")
                    conf = decision.get("confidence")
                    print(f"[signal] {sym} | {act} side={side} conf={conf}")
                    if out.get("analysis_text"):
                        print("[analysis]\n" + out["analysis_text"])
            


            # === META LOG (R:R, ETA, CONF) ===
            meta = out.get("meta", {})
            rr = meta.get("rr", {})
            print(f"[meta] {sym} conf={meta.get('confidence')} rr_min={rr.get('rr_min')} rr_max={rr.get('rr_max')} eta={meta.get('eta')}")
        except Exception:
            print(f"[scan] exception for {sym}:\n{traceback.format_exc()}")

    print(f"[scan] done ts={datetime.utcnow().isoformat()+'Z'} sent={sent}")
    return {"ok": True, "count": sent}

def _scan_loop():
    # vòng lặp nền – chỉ chạy 1 worker
    while True:
        try:
            scan_once_for_logs()
        except Exception:
            print("[scan] loop error:\n" + traceback.format_exc())
        # ngủ theo phút
        time.sleep(max(1, SCAN_INTERVAL_MIN) * 60)

# Khởi động lịch khi app lên
@app.on_event("startup")
def _start_scheduler():
    print(f"[scheduler] start: interval={SCAN_INTERVAL_MIN}min; MAX_GPT={MAX_GPT} USE_1H_TRIGGER={USE_1H_TRIGGER}")
    t = threading.Thread(target=_scan_loop, daemon=True)
    t.start()

# Endpoint để bấm chạy tay 1 lần (trả JSON)
@app.post("/scan_once")
def scan_once_endpoint():
    try:
        res = scan_once_for_logs()
        return {"ok": True, "result": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}


if __name__ == '__main__':
    main()
