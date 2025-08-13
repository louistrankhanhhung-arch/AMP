import argparse, json
from kucoin_api import fetch_batch
from indicators import enrich_indicators, enrich_more
from structure_engine import build_struct_json
from filter import rank_all
from chart_renderer import render_chart

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['onboard','coverage'], default='onboard')
    ap.add_argument('--symbol', default='BTC/USDT')
    ap.add_argument('--tfs', default='4H,1D', help="Comma-separated list of timeframes, e.g. 4H,1D")
    ap.add_argument('--limit', type=int, default=100, help="Number of candles per timeframe")
    args = ap.parse_args()

    tfs = [x.strip().upper() for x in args.tfs.split(',') if x.strip()]
    batch = fetch_batch(symbol, timeframes=["4H", "1D"], limit=100)

    structs = []
    for tf, df in batch.items():
        df = enrich_indicators(df)
        df = enrich_more(df)
        struct = build_struct_json(args.symbol, tf, df)
        st_4h = build_struct_json(symbol, "4H", batch["4H"], context=batch["1D"])
        structs.append(struct)

        # Lưu chart mỗi khung để xem nhanh
        out_img = f'/mnt/data/{args.symbol}_{tf}.png'
        render_chart(df, out_img)
        print(f'Chart for {args.symbol} {tf} saved to', out_img)

    # Xếp hạng & in kết quả
    ranks = rank_all(structs)
    print(json.dumps([r.__dict__ for r in ranks], ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
