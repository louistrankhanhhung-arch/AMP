"""
capture_tv.py
Headless screenshot of TradingView chart with fixed layout (EMA20/50, Bollinger, RSI, Volume)
for given symbol/timeframes. Dùng để lấy ảnh chart “đáng tin cậy” ghép với JSON phân tích.

Yêu cầu:
  pip install playwright
  playwright install chromium

Ví dụ (Git Bash):
  python capture_tv.py --exchange KUCOIN --symbol SUIUSDT --tfs 240,D --outdir out_tv
  # 4H = 240, 1D = D ; cũng hỗ trợ 60 (1H), 120 (2H)
"""

import argparse
import time
from pathlib import Path
from textwrap import dedent

from playwright.sync_api import sync_playwright

TV_JS = "https://s3.tradingview.com/tv.js"

def tf_to_interval(tf: str) -> str:
    tf = tf.upper().strip()
    mapping = {"1H": "60", "2H": "120", "4H": "240", "D": "D", "1D": "D", "DAY": "D"}
    return mapping.get(tf, tf)  # cho phép 60/120/240 trực tiếp

def make_html(exchange: str, symbol: str, interval: str, width=1280, height=720) -> str:
    full_symbol = f"{exchange}:{symbol}"
    # studies: BB(20,2), EMA20, EMA50, RSI(14), Volume
    studies_js = """
      studies: [
        {"id": "BB@tv-basicstudies", "inputs": {"length": 20, "std": 2}},
        {"id": "MAExp@tv-basicstudies", "inputs": {"length": 20}},
        {"id": "MAExp@tv-basicstudies", "inputs": {"length": 50}},
        {"id": "RSI@tv-basicstudies", "inputs": {"length": 14}},
        {"id": "Volume@tv-basicstudies"}
      ],
    """
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <script src="{TV_JS}"></script>
  <style> html,body,#root {{ margin:0; padding:0; background:#fff; }} #root {{ width:{width}px; height:{height}px; }} </style>
</head>
<body>
  <div id="root"></div>
  <script>
    new TradingView.widget({{
      autosize: false,
      width: {width},
      height: {height},
      symbol: "{full_symbol}",
      interval: "{interval}",
      timezone: "Etc/UTC",
      theme: "light",
      style: "1",
      locale: "en",
      hide_top_toolbar: true,
      hide_side_toolbar: true,
      allow_symbol_change: false,
      container_id: "root",
      {studies_js}
    }});
  </script>
</body>
</html>"""
    return html

def capture_once(pw, exchange: str, symbol: str, interval: str, out_path: Path,
                 width=1280, height=720, wait_sec=3):
    html = make_html(exchange, symbol, interval, width=width, height=height)

    browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    try:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
        page.set_content(html, wait_until="load")                 # <— nạp HTML trực tiếp
        page.wait_for_selector("#root iframe", timeout=15000)     # đợi widget render
        time.sleep(wait_sec)                                      # đợi chỉ báo vẽ xong
        page.screenshot(path=str(out_path), full_page=False)
    finally:
        browser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="KUCOIN", help="BINANCE | KUCOIN | BYBIT | OKX ...")
    ap.add_argument("--symbol", required=True, help="Không có dấu '/', ví dụ SUIUSDT, BTCUSDT")
    ap.add_argument("--tfs", default="240,D", help="Danh sách TF: 60,120,240,D hoặc 1H,2H,4H,1D")
    ap.add_argument("--outdir", default="out_tv")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    tfs = [tf_to_interval(x) for x in args.tfs.split(",") if x.strip()]

    with sync_playwright() as pw:
        for tf in tfs:
            fn = outdir / f"{args.symbol.upper()}_{tf}_tv.png"
            capture_once(pw, args.exchange.upper(), args.symbol.upper(), tf, fn,
                         width=args.width, height=args.height)
            print("Saved:", fn)

if __name__ == "__main__":
    main()
