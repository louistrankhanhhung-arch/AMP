import argparse
import json
from datetime import datetime
from typing import List, Dict

# --- your modules ---
from kucoin_api import fetch_batch
from indicators import enrich_indicators, enrich_more, calc_vp, fetch_funding_oi
from structure_engine import build_struct_json
from filter import rank_all
from universe import get_universe_from_env

# ---------------------------
# helpers dùng chung cho CLI & API
# ---------------------------
def _parse_list_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def build_structs_for_symbol(
    symbol: str,
    tfs: List[str],
    limit: int = 300,
    with_futures: bool = False,
    with_liquidity: bool = False,
) -> List[Dict]:
    """Build structs cho 1 symbol với logic: nếu có 1D thì dùng làm context cho 4H."""
    batch = fetch_batch(symbol, timeframes=tfs, limit=limit)

    # context 1D (nếu có) cho 4H
    ctx_df = batch.get("1D")
    if ctx_df is not None:
        ctx_df = enrich_more(enrich_indicators(ctx_df))

    futures_sent = fetch_funding_oi(symbol) if with_futures else None

    out = []
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
# CLI mode (giữ nguyên hành vi cũ)
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

    tfs = [x.strip().upper() for x in args.tfs.split(',') if x.strip()]
    structs = build_structs_for_symbol(
        args.symbol, tfs, limit=args.limit,
        with_futures=args.with_futures,
        with_liquidity=args.with_liquidity
    )
    # Xếp hạng & in kết quả (như cũ)
    ranks = rank_all(structs)
    print(json.dumps([r.__dict__ for r in ranks], ensure_ascii=False, indent=2))

# ---------------------------
# FastAPI app cho Railway
# ---------------------------
try:
    from fastapi import FastAPI, Query
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "ts": datetime.utcnow().isoformat() + "Z"}

    @app.get("/structs.json")
    def structs(symbols: str = "", tfs: str = "4H,1D", limit: int = 300, with_futures: int = 0, with_liquidity: int = 0):
        syms = [s.strip() for s in symbols.split(",") if s.strip()] or get_universe_from_env()
        tflist = [x.strip().upper() for x in tfs.split(",") if x.strip()]
        out = []
        for sym in syms:
            out.extend(build_structs_for_symbol(
                sym, tflist, limit=limit,
                with_futures=bool(with_futures),
                with_liquidity=bool(with_liquidity)
            ))
        return {"generated_at": datetime.utcnow().isoformat()+"Z", "structs": out}


    @app.get("/ranks.json")
    def ranks(symbols: str = "", tfs: str = "4H,1D", limit: int = 300, with_futures: int = 0, with_liquidity: int = 0):
        syms = [s.strip() for s in symbols.split(",") if s.strip()] or get_universe_from_env()
        tflist = [x.strip().upper() for x in tfs.split(",") if x.strip()]
        all_structs = []
        for sym in syms:
            all_structs.extend(build_structs_for_symbol(
                sym, tflist, limit=limit,
                with_futures=bool(with_futures),
                with_liquidity=bool(with_liquidity)
            ))
        rks = rank_all(all_structs)
        return {"generated_at": datetime.utcnow().isoformat()+"Z","ranks":[r.__dict__ for r in rks]}


    @app.get("/bucketA_structs.json")
    def bucketA_structs(symbols: str = "", tfs: str = "4H,1D", limit: int = 300,
                        with_futures: int = 0, with_liquidity: int = 0,
                        min_bucket: str = "A", min_score: float = 7.0):
        syms = [s.strip() for s in symbols.split(",") if s.strip()] or get_universe_from_env()
        tflist = [x.strip().upper() for x in tfs.split(",") if x.strip()]
    
        all_structs = []
        for sym in syms:
            all_structs.extend(build_structs_for_symbol(
                sym, tflist, limit=limit,
                with_futures=bool(with_futures),
                with_liquidity=bool(with_liquidity)
            ))
    
        rks = rank_all(all_structs)
    
        def _ord(b): return {"A":3,"B":2,"C":1}.get((b or "C").upper(),0)
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



        # chọn các symbol đạt filter dựa trên 4H (rank_all dùng toàn bộ structs)
        rks = rank_all(all_structs)
        def _ord(b): return {"A":3,"B":2,"C":1}.get((b or "C").upper(),0)
        ok_syms = set()
        for r in rks:
            bucket = getattr(r, "bucket_best", None) or getattr(r, "bucket", None)
            score  = getattr(r, "score_best", None)  or getattr(r, "score", None)
            if (bucket and _ord(bucket) >= _ord(min_bucket)) or (isinstance(score,(int,float)) and score >= float(min_score)):
                ok_syms.add(r.symbol)

        filtered = [s for s in all_structs if s.get("symbol") in ok_syms]
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "symbols": sorted(ok_syms),
            "structs": filtered,
            "ranks": [r.__dict__ for r in rks if r.symbol in ok_syms]
        }

except Exception:
    # Nếu FastAPI không có (chạy CLI thuần), bỏ qua.
    app = None

if __name__ == '__main__':
    main()
