
"""
render_panels.py
Render 3-pane charts (Price/Volume/RSI) with overlays from raw OHLCV for given symbol & timeframes.

Layout:
  (1) Price + EMA20/50 + Bollinger Bands + swing pivots (HH/LL) + SR (up/down) + Fibonacci of latest leg + Liquidity zones (if computed)
  (2) Volume + SMA20
  (3) RSI(14) + 30/50/70 lines

Notes:
- Uses matplotlib only. If mplfinance is available, it will draw candlesticks; otherwise close-price line.
- Does not rely on the JSON; it fetches and enriches OHLCV directly (consistent with Step 1 in your pipeline).
- Designed to be run before sending charts + JSON to GPT.

Usage (Git Bash):
  python render_panels.py --symbol SUI/USDT --tfs 4H,1D --limit 300 --outdir out_charts --with-liquidity
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import mplfinance as mpf
    HAS_MPLFIN = True
except Exception:
    HAS_MPLFIN = False

from kucoin_api import fetch_batch
from indicators import enrich_indicators, enrich_more, calc_vp
from structure_engine import find_swings, find_sr

# --------------- helpers ---------------

def _fibo_from_swings(swings: List[Dict]) -> Optional[Dict[str, List[float]]]:
    """
    Pick the latest leg using the last two swings: if last is HH use prev as low, else if last is LL use prev as high.
    Return dict {"leg": [low, high], "levels": [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1] *on price*}
    """
    if len(swings) < 2:
        return None
    last = swings[-1]
    prev = swings[-2]
    typ = last.get("type")
    p_last = float(last.get("price"))
    p_prev = float(prev.get("price"))
    if typ == "HH":
        lo, hi = (min(p_prev, p_last), max(p_prev, p_last))
    else:  # LL or others
        lo, hi = (min(p_prev, p_last), max(p_prev, p_last))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = [round(hi - (hi - lo) * r, 4) for r in ratios]
    return {"leg": [lo, hi], "levels": levels, "ratios": ratios}

def _draw_sr(ax, sr_levels: List[float], xmin, xmax, linestyle='--'):
    if not sr_levels:
        return
    for lv in sr_levels:
        try:
            ax.hlines(lv, xmin, xmax, linestyles=linestyle, linewidth=0.8)
            ax.text(xmax, lv, f"{lv:.2f}", va="center", ha="right", fontsize=8)
        except Exception:
            continue

def _draw_liquidity(ax, zones, xmin, xmax):
    if not zones:
        return
    for z in zones:
        lo, hi = z.get("price_range", (None, None))
        if lo is None or hi is None:
            continue
        lo, hi = float(lo), float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        ax.axhspan(lo, hi, alpha=0.08)  # light highlight

def _scatter_swings(ax, df: pd.DataFrame, swings: List[Dict]):
    if not swings:
        return
    for s in swings[-30:]:  # limit labels
        t = pd.to_datetime(s["t"])
        p = float(s["price"])
        ax.scatter([t], [p], s=12, zorder=3)
        try:
            ax.text(t, p, s["type"], fontsize=7, va="bottom", ha="center")
        except Exception:
            pass

def _plot_price_panel(ax, df: pd.DataFrame, swings: List[Dict], sr_up: List[float], sr_down: List[float], lz=None):
    # Price: candlesticks if available, else close line
    if HAS_MPLFIN:
        mdf = df[["open","high","low","close","volume"]].copy()
        mdf.index.name = "Date"
        mpf.plot(mdf, type='candle', ax=ax, axtitle="", volume=False, style="default", mav=(20,50))
        # Bollinger from df (if present)
        if {"bb_upper","bb_mid","bb_lower"}.issubset(df.columns):
            ax.plot(df.index, df["bb_upper"])
            ax.plot(df.index, df["bb_mid"])
            ax.plot(df.index, df["bb_lower"])
    else:
        ax.plot(df.index, df["close"], linewidth=1.0)
        if "ema20" in df.columns and "ema50" in df.columns:
            ax.plot(df.index, df["ema20"])
            ax.plot(df.index, df["ema50"])
        if {"bb_upper","bb_mid","bb_lower"}.issubset(df.columns):
            ax.plot(df.index, df["bb_upper"])
            ax.plot(df.index, df["bb_mid"])
            ax.plot(df.index, df["bb_lower"])

    # Overlays
    xmin, xmax = df.index.min(), df.index.max()
    _scatter_swings(ax, df, swings)
    _draw_sr(ax, sr_up or [], xmin, xmax, linestyle='--')
    _draw_sr(ax, sr_down or [], xmin, xmax, linestyle=':')
    if lz:
        _draw_liquidity(ax, lz, xmin, xmax)

    # Fibonacci
    fib = _fibo_from_swings(swings)
    if fib:
        for lv in fib["levels"]:
            ax.hlines(lv, xmin, xmax, linewidth=0.8)

    ax.set_ylabel("Price")
    ax.grid(True, linewidth=0.3)

def _plot_volume_panel(ax, df: pd.DataFrame):
    ax.bar(df.index, df["volume"], width=0.8)
    if "vol_sma20" in df.columns:
        ax.plot(df.index, df["vol_sma20"])
    ax.set_ylabel("Volume")
    ax.grid(True, linewidth=0.3)

def _plot_rsi_panel(ax, df: pd.DataFrame):
    if "rsi14" in df.columns:
        ax.plot(df.index, df["rsi14"])
    ax.axhline(70, linewidth=0.8)
    ax.axhline(50, linewidth=0.8)
    ax.axhline(30, linewidth=0.8)
    ax.set_ylabel("RSI14")
    ax.grid(True, linewidth=0.3)

# --------------- main ---------------

def render_panels_for_tf(symbol: str, tf: str, df: pd.DataFrame, out_path: Path, with_liquidity: bool):
    # Ensure indicators
    df = enrich_more(enrich_indicators(df))

    # Compute overlays
    swings = find_swings(df)
    sr = find_sr(df, swings=None)  # find_sr doesn't use swings internally
    lz = calc_vp(df) if with_liquidity else None

    # Figure
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[3,1,1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    _plot_price_panel(ax1, df, swings, sr.get("sr_up"), sr.get("sr_down"), lz)
    _plot_volume_panel(ax2, df)
    _plot_rsi_panel(ax3, df)

    fig.suptitle(f"{symbol} {tf}", y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--tfs", default="4H,1D", help="Comma-separated timeframes, e.g. 4H,1D")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--outdir", default="out_charts")
    ap.add_argument("--with-liquidity", action="store_true")
    args = ap.parse_args()

    tfs = [x.strip().upper() for x in args.tfs.split(",") if x.strip()]
    batch = fetch_batch(args.symbol, timeframes=tfs, limit=args.limit)

    for tf, df in batch.items():
        out_img = Path(args.outdir) / f"{args.symbol.replace('/','')}_{tf}_panels.png"
        render_panels_for_tf(args.symbol, tf, df, out_img, with_liquidity=args.with_liquidity)
        print("Saved:", out_img)

if __name__ == "__main__":
    main()
