import argparse, json
from app.config import settings
from app.kucoin_api import fetch_ohlcv
from app.indicators import enrich_indicators
from app.structure_engine import build_struct_json
from app.filter import rank_all
from app.chart_renderer import render_chart

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['onboard','coverage'], default='onboard')
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--tf', default='4H')
    args = ap.parse_args()

    df = fetch_ohlcv(args.symbol, args.tf, limit=300)
    df = enrich_indicators(df)
    struct = build_struct_json(args.symbol, args.tf, df)
    ranks = rank_all([struct])
    print(json.dumps([r.__dict__ for r in ranks], ensure_ascii=False, indent=2))

    out_img = '/mnt/data/preview.png'
    render_chart(df, out_img)
    print('Chart saved to', out_img)

if __name__ == '__main__':
    main()
