"""
batch_triggers.py
Run 1H triggers (LONG/SHORT) for a set of symbols that passed 4H/1D filter.
Optionally capture TradingView images for multiple timeframes (e.g., 60,240,D).

Inputs (choose one):
  --symbols        e.g. "SUI/USDT,BTC/USDT"
  --structs-json   path to JSON from dump_structs.py (expects {"structs":[...]}).

Options:
  --exchange       Exchange prefix for TradingView capture (BINANCE/KUCOIN/...) [default: KUCOIN]
  --capture        Capture TradingView images for each symbol
  --capture-tfs    Comma-separated intervals to capture (default: 60,240,D). Supports 60/120/240/D or 1H/2H/4H/1D
  --min-bucket     Minimum bucket to include (A/B/C) [default: A]
  --min-score      Minimum score to include (fallback if bucket missing) [default: 7.0]
  --limit          Candles to fetch for 1H struct (default: 300)
  --out            Output folder [default: out_batch_triggers]

Usage:
  python batch_triggers.py --structs-json out/SOME.json --capture --exchange KUCOIN --capture-tfs 60,240,D
  python batch_triggers.py --symbols SUI/USDT,BTC/USDT --capture --exchange BINANCE --capture-tfs D,240,60
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Set

from kucoin_api import fetch_batch
from indicators import enrich_indicators, enrich_more
from structure_engine import build_struct_json
from trigger_1h import check_long_trigger, check_short_trigger

def _order_bucket(b: str) -> int:
    order = {"A": 3, "B": 2, "C": 1}
    return order.get((b or "C").upper(), 0)

def _pick_bucket_from_structs(structs: List[Dict], min_bucket: str = "A", min_score: float = 7.0) -> Set[str]:
    """Pick symbols meeting min_bucket/min_score using filter.score_symbol on 4H structs."""
    from filter import score_symbol  # lazy import so 'filter.py' stays optional for other tasks
    chosen: Set[str] = set()

    for s in structs or []:
        tf = (s.get("timeframe") or "").upper()
        if tf != "4H":
            continue
        sym = s.get("symbol")
        try:
            res = score_symbol(s)
        except Exception:
            continue

        bucket = getattr(res, "bucket_best", None) or getattr(res, "bucket", None)
        score = getattr(res, "score_best", None) or getattr(res, "score", None)

        ok = False
        if bucket and _order_bucket(bucket) >= _order_bucket(min_bucket):
            ok = True
        elif isinstance(score, (int, float)) and float(score) >= float(min_score):
            ok = True

        if ok and sym:
            chosen.add(sym)
    return chosen

def _fetch_struct_1h(symbol: str, limit: int = 300) -> Dict[str, Any]:
    batch = fetch_batch(symbol, timeframes=["1H"], limit=limit)
    df = batch.get("1H")
    df = enrich_more(enrich_indicators(df))
    s1h = build_struct_json(symbol, "1H", df, context_df=None, liquidity_zones=None, futures_sentiment=None)
    return s1h

def _normalize_tf(tf: str) -> str:
    tf = (tf or "").strip().upper()
    mapping = {"1H": "60", "2H": "120", "4H": "240", "DAY": "D", "1D": "D", "D": "D", "60": "60", "120": "120", "240": "240"}
    return mapping.get(tf, tf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="Comma-separated spot symbols with slash, e.g. SUI/USDT,BTC/USDT")
    ap.add_argument("--structs-json", default="", help="Path to JSON file containing {'structs': [...]}")
    ap.add_argument("--exchange", default="KUCOIN", help="BINANCE | KUCOIN | BYBIT | OKX ...")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--capture-tfs", default="60,240,D", help="Intervals to capture (e.g. 60,240,D or 1H,4H,1D)")
    ap.add_argument("--min-bucket", default="A")
    ap.add_argument("--min-score", type=float, default=7.0)
    ap.add_argument("--out", default="out_batch_triggers")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Build symbol set
    symbols: Set[str] = set()
    if args.symbols.strip():
        for x in args.symbols.split(","):
            x = x.strip()
            if x:
                symbols.add(x)
    elif args.structs_json:
        data = json.loads(Path(args.structs_json).read_text(encoding="utf-8"))
        structs = data.get("structs") or []
        symbols = _pick_bucket_from_structs(structs, min_bucket=args.min_bucket, min_score=args.min_score)
    else:
        print("Please provide --symbols or --structs-json")
        return

    # Optional capture imports (load lazily so app can run without playwright)
    HAS_CAPTURE = False
    cap_tfs: List[str] = []
    if args.capture:
        try:
            from capture_tv import tf_to_interval, capture_once
            from playwright.sync_api import sync_playwright
            HAS_CAPTURE = True
            cap_tfs = [_normalize_tf(tf) for tf in args.capture_tfs.split(",") if tf.strip()]
        except Exception as e:
            print(f"capture_tv not available; skip images. ({e})")
            HAS_CAPTURE = False

    summary = []

    # If capturing, keep a single browser session for all symbols (faster)
    if HAS_CAPTURE:
        from capture_tv import capture_once  # type: ignore
        from playwright.sync_api import sync_playwright  # type: ignore
        with sync_playwright() as pw:
            for sym in sorted(symbols):
                try:
                    s1h = _fetch_struct_1h(sym, limit=args.limit)
                    long_trig = check_long_trigger(s1h)
                    short_trig = check_short_trigger(s1h)

                    img_map: Dict[str, str] = {}
                    sym_tv = sym.replace("/", "")
                    for tf in cap_tfs:
                        fn = outdir / f"{sym_tv}_{tf}_tv.png"
                        interval = tf if tf in ("60","120","240","D") else _normalize_tf(tf)
                        capture_once(pw, args.exchange.upper(), sym_tv.upper(), interval, fn)
                        img_map[tf] = str(fn)

                    row = {
                        "symbol": sym,
                        "timeframe": "1H",
                        "trigger_long": long_trig,
                        "trigger_short": short_trig,
                        "images": img_map if img_map else None
                    }
                    summary.append(row)
                    print(f"{sym}: long={long_trig.get('ok')} ({long_trig.get('type')}), short={short_trig.get('ok')} ({short_trig.get('type')})")
                except Exception as e:
                    summary.append({"symbol": sym, "error": str(e)})
                    print(f"{sym}: ERROR {e}")
    else:
        # No capture: still compute triggers
        for sym in sorted(symbols):
            try:
                s1h = _fetch_struct_1h(sym, limit=args.limit)
                long_trig = check_long_trigger(s1h)
                short_trig = check_short_trigger(s1h)
                row = {
                    "symbol": sym,
                    "timeframe": "1H",
                    "trigger_long": long_trig,
                    "trigger_short": short_trig,
                    "images": None
                }
                summary.append(row)
                print(f"{sym}: long={long_trig.get('ok')} ({long_trig.get('type')}), short={short_trig.get('ok')} ({short_trig.get('type')})")
            except Exception as e:
                summary.append({"symbol": sym, "error": str(e)})
                print(f"{sym}: ERROR {e}")

    # Save summary
    out_file = outdir / "batch_triggers_1h.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
