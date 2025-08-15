# structure_engine.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import math

# =========================
# Helpers
# =========================

def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _last_closed_idx(df) -> int:
    """
    Trả về index của nến ĐÃ ĐÓNG gần nhất.
    - Nếu DataFrame có >= 2 hàng: dùng -2 (vì hàng cuối có thể là nến đang chạy)
    - Nếu chỉ có 1 hàng: fallback -1 (dev/test)
    """
    return -2 if len(df) >= 2 else -1

def _get(df, col, i, default=None, cast=float):
    try:
        if col in df:
            v = df[col].iloc[i]
            return None if v is None else cast(v)
    except Exception:
        pass
    return default

def _safe_minmax(a: float, b: float, fn=min):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return fn(a, b)

# =========================
# Events & structure parts
# =========================

def candle_flags(df) -> Dict[str, Any]:
    """
    Phát hiện một vài mẫu nến cơ bản (PIN/ENGULF) dựa trên NẾN ĐÃ ĐÓNG.
    """
    if len(df) < 3:
        return {
            "bullish_engulf": False,
            "bearish_engulf": False,
            "pin_bull": False,
            "pin_bear": False,
        }

    i = _last_closed_idx(df)            # nến đã đóng
    prev = i - 1                        # nến trước đó (cũng đã đóng)
    o1, h1, l1, c1 = float(df["open"].iloc[prev]), float(df["high"].iloc[prev]), float(df["low"].iloc[prev]), float(df["close"].iloc[prev])
    o2, h2, l2, c2 = float(df["open"].iloc[i]),    float(df["high"].iloc[i]),    float(df["low"].iloc[i]),    float(df["close"].iloc[i])

    # Engulf: thân nến 2 bao trùm thân nến 1
    bull_engulf = (c2 > o2) and (c1 < o1) and (o2 <= c1) and (c2 >= o1)
    bear_engulf = (c2 < o2) and (c1 > o1) and (o2 >= c1) and (c2 <= o1)

    # Pin bar: bóng dài vượt trội
    body2 = abs(c2 - o2)
    upper = h2 - max(c2, o2)
    lower = min(c2, o2) - l2
    # tiêu chí tương đối
    pin_bull = (lower > body2 * 2.0) and (lower > upper * 1.2)
    pin_bear = (upper > body2 * 2.0) and (upper > lower * 1.2)

    return {
        "bullish_engulf": bool(bull_engulf),
        "bearish_engulf": bool(bear_engulf),
        "pin_bull": bool(pin_bull),
        "pin_bear": bool(pin_bear),
    }


def detect_trend(df, ctx_sw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Trend đơn giản dựa trên EMA20/EMA50 và vị trí close — TẤT CẢ dùng nến ĐÃ ĐÓNG.
    """
    i = _last_closed_idx(df)
    ema20 = _get(df, "ema20", i)
    ema50 = _get(df, "ema50", i)
    close = _get(df, "close", i)

    trend_up = False
    trend_down = False
    ema_cross_up = None
    ema_cross_down = None

    if ema20 is not None and ema50 is not None:
        if ema20 > ema50:
            trend_up = True
            ema_cross_up = True
            ema_cross_down = False
        elif ema20 < ema50:
            trend_down = True
            ema_cross_up = False
            ema_cross_down = True

    if close is not None and ema20 is not None:
        # position vs EMA20 (bám biên)
        above_ema20 = close > ema20
    else:
        above_ema20 = None

    return {
        "trend_up": bool(trend_up),
        "trend_down": bool(trend_down),
        "ema20_gt_ema50": True if (ema20 is not None and ema50 is not None and ema20 > ema50) else False,
        "above_ema20": above_ema20,
        "ema_cross_up": ema_cross_up,
        "ema_cross_down": ema_cross_down,
    }


def detect_breakout(df, lookback: int = 20) -> Dict[str, Any]:
    """
    Xác nhận breakout/breakdown dựa trên NẾN ĐÃ ĐÓNG gần nhất.
    - breakout: close_last > prior_high (20 nến trước nến đã đóng)
    - breakdown: close_last < prior_low
    """
    if len(df) < max(lookback, 3):
        return {
            "last_breakout_confirmed": False,
            "last_breakout_level": None,
            "last_breakdown_confirmed": False,
            "last_breakdown_level": None,
        }

    i = _last_closed_idx(df)
    last_close = float(df["close"].iloc[i])

    # cửa sổ trước nến đã đóng gần nhất
    end = len(df) + i  # i=-2 -> end=len-2
    start = max(0, end - lookback)
    win = df.iloc[start:end]

    prior_high = float(win["high"].max()) if len(win) else float(df["high"].iloc[i])
    prior_low  = float(win["low"].min())  if len(win) else float(df["low"].iloc[i])

    breakout_up  = last_close > prior_high
    breakdown_dn = last_close < prior_low

    return {
        "last_breakout_confirmed": bool(breakout_up),
        "last_breakout_level": prior_high if breakout_up else None,
        "last_breakdown_confirmed": bool(breakdown_dn),
        "last_breakdown_level": prior_low if breakdown_dn else None,
    }


def detect_market_structure(df, swing_lookback: int = 5) -> Dict[str, Any]:
    """
    Cấu trúc đỉnh-đáy cơ bản (HH/HL/LH/LL) trên NẾN ĐÃ ĐÓNG.
    Trả nhãn gợi ý: 'bullish_continuation' | 'bearish_continuation' | 'sideway'
    """
    if len(df) < swing_lookback + 3:
        return {"label": "sideway", "last_swings": []}

    i = _last_closed_idx(df)
    # đơn giản: so sánh close với EMA20/50 và độ dốc EMA20
    ema20_now = _get(df, "ema20", i)
    ema20_prev = _get(df, "ema20", i - 1)
    ema50_now = _get(df, "ema50", i)

    label = "sideway"
    if (ema20_now is not None and ema50_now is not None and ema20_prev is not None):
        if ema20_now > ema50_now and ema20_now >= ema20_prev:
            label = "bullish_continuation"
        elif ema20_now < ema50_now and ema20_now <= ema20_prev:
            label = "bearish_continuation"

    return {"label": label, "last_swings": []}


def extract_context_levels(df) -> Dict[str, Any]:
    """
    Suy diễn vài mức SR gần (mềm) để GPT tham khảo.
    (Nếu bạn đã có logic riêng, cứ giữ/merge thêm trường này.)
    """
    if len(df) < 10:
        return {"soft_support": None, "soft_resistance": None, "sr_up": None, "sr_down": None}

    i = _last_closed_idx(df)
    ema20 = _get(df, "ema20", i)
    ema50 = _get(df, "ema50", i)
    bb_up = _get(df, "bb_up", i)
    bb_low = _get(df, "bb_low", i)

    soft_support    = _safe_minmax(ema20, ema50, min)
    soft_resistance = _safe_minmax(bb_up, ema50, max)

    # dải SR rộng hơn (tham khảo)
    highN = float(df["high"].iloc[max(0, len(df)-50):len(df)+i].max()) if len(df) >= 2 else float(df["high"].iloc[i])
    lowN  = float(df["low"].iloc[max(0, len(df)-50):len(df)+i].min())  if len(df) >= 2 else float(df["low"].iloc[i])

    return {
        "soft_support": soft_support,
        "soft_resistance": soft_resistance,
        "sr_up": (soft_resistance, highN),
        "sr_down": (soft_support, lowN),
    }

# =========================
# Main builder
# =========================

def build_struct_json(
    symbol: str,
    timeframe: str,
    df, *,
    context_df=None,
    liquidity_zones: Optional[Dict[str, Any]] = None,
    futures_sentiment: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Trả về cấu trúc JSON dùng cho filter/GPT.
    TẤT CẢ số liệu cuối đều dựa trên NẾN ĐÃ ĐÓNG.
    """
    tf = timeframe.upper()
    i  = _last_closed_idx(df)

    snapshot = {
        "ts":       int(df["timestamp"].iloc[i]),
        "open":     float(df["open"].iloc[i]),
        "high":     float(df["high"].iloc[i]),
        "low":      float(df["low"].iloc[i]),
        "close":    float(df["close"].iloc[i]),
        "ema20":    _get(df, "ema20", i),
        "ema50":    _get(df, "ema50", i),
        "bb_mid":   _get(df, "bb_mid", i),
        "bb_up":    _get(df, "bb_up", i),
        "bb_low":   _get(df, "bb_low", i),
        "rsi":      _get(df, "rsi", i),
        "atr":      _get(df, "atr", i),
        "volume":   _get(df, "volume", i),
    }

    # events (đều dựa trên nến đã đóng)
    ev = {}
    ev.update(candle_flags(df))
    ev.update(detect_breakout(df))
    ms = detect_market_structure(df)

    # trend hiện tại
    tr = detect_trend(df, ctx_sw=None)

    # context levels (theo TF hiện tại; nếu cần bạn có thể tách logic 1D khác 4H)
    ctx_lv = extract_context_levels(context_df if (tf == "4H" and context_df is not None) else df)

    # stats phụ (tuỳ DF có cột gì)
    stats = {
        "atr14": snapshot["atr"],
        "rsi14": snapshot["rsi"],
    }

    out = {
        "generated_at": _utcnow(),
        "symbol": symbol,
        "timeframe": tf,
        "snapshot": snapshot,      # CHỐT số liệu ở nến đã đóng
        "trend": tr,
        "market_structure": ms,
        "events": ev,              # CÓ: last_breakout_confirmed, last_breakdown_confirmed
        "context_levels": ctx_lv,
        "stats": stats,
        "liquidity": liquidity_zones or {},
        "futures": futures_sentiment or {},
        # tương thích ngược (nếu đâu đó dùng 'last'):
        "last": snapshot,
    }
    return out
