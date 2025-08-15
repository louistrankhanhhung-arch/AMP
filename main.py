import argparse
import json
from datetime import datetime
from typing import List, Dict, Optional

# --- your modules ---
from kucoin_api import fetch_batch
from indicators import enrich_indicators, enrich_more, calc_vp, fetch_funding_oi
from structure_engine import build_struct_json
from filter import rank_all
from universe import resolve_symbols  # <= dùng hàm chuẩn hoá danh mục mã
from gpt_signal_builder import make_telegram_signal

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
from fastapi import FastAPI

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
    structs = build_structs_for_symbol(symbol, tflist, limit=limit, with_futures=bool(with_futures), with_liquidity=bool(with_liquidity))
    s4 = next((s for s in structs if s.get("timeframe")=="4H"), None)
    s1 = next((s for s in structs if s.get("timeframe")=="1D"), None)
    if not s4 or not s1:
        return {"ok": False, "error": "missing 4H or 1D struct"}

    out = make_telegram_signal(s4, s1, trigger_1h=None)  # nếu có trigger_1H JSON, truyền vào đây
    return out

# ====== Background scanner writing signals to logs ======
import os, time, threading, traceback
from universe import resolve_symbols
from gpt_signal_builder import make_telegram_signal

SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "60"))
SCAN_TFS          = os.getenv("SCAN_TFS", "4H,1D")
SCAN_LIMIT        = int(os.getenv("SCAN_LIMIT", "200"))
MIN_BUCKET        = os.getenv("MIN_BUCKET", "A")
MIN_SCORE         = float(os.getenv("MIN_SCORE", "7"))
MAX_GPT           = int(os.getenv("MAX_GPT", "8"))

def _bucket_ord(b: str) -> int:
    return {"A":3, "B":2, "C":1}.get((b or "C").upper(), 0)

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

    # 3) Gọi GPT cho tối đa MAX_GPT mã
    picked = [s for s in syms if s in ok_syms][:MAX_GPT]
    print(f"[scan] candidates={list(picked)} (cap {MAX_GPT})")

    sent = 0
    for sym in picked:
        try:
            s4 = next((x for x in all_structs if x.get("symbol")==sym and x.get("timeframe")=="4H"), None)
            s1 = next((x for x in all_structs if x.get("symbol")==sym and x.get("timeframe")=="1D"), None)
            if not s4 or not s1:
                print(f"[scan] missing structs: {sym}")
                continue

            out = make_telegram_signal(s4, s1, trigger_1h=None)  # nếu sau dùng trigger_1H thì truyền vào đây
            if not out.get("ok"):
                print(f"[scan] GPT err {sym}: {out.get('error')}")
                continue

            tele = out.get("telegram_text")
            decision = out.get("decision", {})
            plan = out.get("plan", {})

            if tele:
                # === KẾT QUẢ GỬI TELEGRAM – IN THẲNG RA LOG ===
                print("[signal]\n" + tele)
                sent += 1
            else:
                # WAIT/AVOID
                act = (decision.get("action") or "").upper()
                side = decision.get("side")
                conf = decision.get("confidence")
                print(f"[signal] {sym} | {act} side={side} conf={conf}")

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
    print(f"[scheduler] start: interval={SCAN_INTERVAL_MIN}min; MAX_GPT={MAX_GPT}")
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

python -m uvicorn main:app --host :: --port $PORT --workers 1

if __name__ == '__main__':
    main()
