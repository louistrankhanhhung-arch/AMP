
import argparse
import json
from pathlib import Path

from kucoin_api import fetch_batch
from indicators import enrich_indicators, enrich_more
from structure_engine import build_struct_json
from trigger_1h import check_long_trigger, check_short_trigger

try:
    from capture_tv import tf_to_interval, capture_once
    from playwright.sync_api import sync_playwright
    HAS_CAPTURE = True
except Exception:
    HAS_CAPTURE = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="KUCOIN")
    ap.add_argument("--symbol", required=True, help="Spot symbol with slash, e.g., SUI/USDT")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--capture", action="store_true", help="Capture 1H image via TradingView")
    ap.add_argument("--out", default="out_triggers")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    batch = fetch_batch(args.symbol, timeframes=["1H"], limit=args.limit)
    df = batch.get("1H")
    df = enrich_more(enrich_indicators(df))

    s1h = build_struct_json(args.symbol, "1H", df, context_df=None, liquidity_zones=None, futures_sentiment=None)

    long_trig = check_long_trigger(s1h)
    short_trig = check_short_trigger(s1h)

    img_path = None
    if args.capture and HAS_CAPTURE:
        sym_tv = args.symbol.replace("/", "")
        img_path = outdir / f"{sym_tv}_60_tv.png"
        with sync_playwright() as pw:
            capture_once(pw, args.exchange.upper(), sym_tv.upper(), tf_to_interval("1H"), img_path)

    out_obj = {
        "symbol": args.symbol,
        "timeframe": "1H",
        "trigger_long": long_trig,
        "trigger_short": short_trig,
        "image_1h": str(img_path) if img_path else None
    }
    out_file = outdir / f"{args.symbol.replace('/','')}_1H_trigger.json"
    out_file.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_file}")
    if img_path:
        print(f"Image: {img_path}")

if __name__ == "__main__":
    main()
