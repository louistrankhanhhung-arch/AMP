import argparse
import json

from kucoin_api import fetch_batch
from indicators import enrich_indicators, enrich_more, calc_vp, fetch_funding_oi
from structure_engine import build_struct_json
from filter import rank_all
from chart_renderer import render_chart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['onboard', 'coverage'], default='onboard')
    ap.add_argument('--symbol', default='BTC/USDT')
    ap.add_argument('--tfs', default='4H,1D', help='Comma-separated, e.g. 4H,1D')
    ap.add_argument('--limit', type=int, default=100, help='Candles per timeframe')
    ap.add_argument('--charts', action='store_true', help='Save charts for each timeframe')
    ap.add_argument('--with-futures', action='store_true', help='Attach funding/OI from KuCoin Futures')
    ap.add_argument('--with-liquidity', action='store_true', help='Attach liquidity zones (volume profile)')
    args = ap.parse_args()

    # parse timeframes từ tham số
    tfs = [x.strip().upper() for x in args.tfs.split(',') if x.strip()]

    # lấy dữ liệu theo đúng tham số
    batch = fetch_batch(args.symbol, timeframes=tfs, limit=args.limit)

    # nếu có cả 4H và 1D, dùng 1D làm context cho 4H
    ctx_df = batch.get('1D')
    if ctx_df is not None:
        # Ensure context_df has indicators for detect_trend/find_sr/etc.
        ctx_df = enrich_more(enrich_indicators(ctx_df))

    # optional dữ liệu bổ sung (dùng lại cho mọi tf để đỡ gọi lặp)
    futures_sent = fetch_funding_oi(args.symbol) if args.with_futures else None

    structs = []
    for tf, df in batch.items():
        # enrich đầy đủ chỉ báo
        df = enrich_more(enrich_indicators(df))

        # optional: liquidity zones tính trên chính df tf đó
        lz = calc_vp(df) if args.with_liquidity else None

        # chọn context_df: nếu đang ở 4H và có 1D, gắn context 1D; ngược lại None
        context_df = ctx_df if (tf == '4H' and ctx_df is not None) else None

        # build struct
        struct = build_struct_json(
            args.symbol, tf, df,
            context_df=context_df,
            liquidity_zones=lz,
            futures_sentiment=futures_sent
        )
        structs.append(struct)

        # vẽ chart nếu bật flag
        if args.charts:
            out_img = f'{args.symbol.replace("/", "")}_{tf}.png'
            render_chart(df, out_img)
            print(f'Chart for {args.symbol} {tf} saved to', out_img)

    # Xếp hạng & in kết quả
    ranks = rank_all(structs)
    print(json.dumps([r.__dict__ for r in ranks], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
