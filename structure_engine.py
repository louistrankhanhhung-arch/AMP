# structure_engine.py
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

# -------------------------
# 1) Zigzag & Swings (đơn giản)
# -------------------------
def _zigzag(series: pd.Series, pct: float = 2.0) -> List[Tuple[pd.Timestamp, float]]:
    pts = []
    if series.empty:
        return pts
    last_ext = series.iloc[0]
    last_t = series.index[0]
    direction = 0  # 1 up, -1 down, 0 none
    for t, v in series.items():
        change_pct = (v - last_ext) / last_ext * 100 if last_ext != 0 else 0
        if direction >= 0 and change_pct >= pct:
            pts.append((last_t, last_ext))
            last_ext, last_t, direction = v, t, 1
        elif direction <= 0 and change_pct <= -pct:
            pts.append((last_t, last_ext))
            last_ext, last_t, direction = v, t, -1
        else:
            if (direction >= 0 and v > last_ext) or (direction <= 0 and v < last_ext):
                last_ext, last_t = v, t
    pts.append((last_t, last_ext))
    return pts

def find_swings(df: pd.DataFrame, zigzag_pct: float = 2.0):
    zz = _zigzag(df['close'], zigzag_pct)
    out = []
    for i in range(1, len(zz)):
        prev, curr = zz[i-1][1], zz[i][1]
        t = zz[i][0]
        out.append({"type": "HH" if curr > prev else "LL", "t": str(t), "price": float(curr)})
    return out[-20:]

def classify_market_structure_from_swings(swings: List[Dict[str, Any]]) -> List[str]:
    """Suy luận đơn giản từ chuỗi HH/LL (không dùng HL/LH)."""
    tags: List[str] = []
    for i in range(1, len(swings)):
        if swings[i]["type"] == "HH" and swings[i-1]["type"] == "HH" and swings[i]["price"] > swings[i-1]["price"]:
            tags.append("bullish_continuation")
        if swings[i]["type"] == "LL" and swings[i-1]["type"] == "LL" and swings[i]["price"] < swings[i-1]["price"]:
            tags.append("bearish_continuation")
    return tags[-3:]

# -------------------------
# 2) Trend / SR / Pullback / Divergence
# -------------------------
def detect_trend(df: pd.DataFrame, swings) -> Dict[str, Any]:
    ema20, ema50 = df['ema20'].iloc[-1], df['ema50'].iloc[-1]
    state = "up" if ema20 > ema50 else ("down" if ema20 < ema50 else "side")
    age = min(len(df), 100)
    return {"state": state, "basis": "ema20 vs ema50", "age_bars": age}

def find_sr(df: pd.DataFrame, swings) -> Dict[str, list]:
    closes = df['close'].tail(200)
    sr_up = sorted({round(x, 2) for x in closes.nlargest(6).tolist()})
    sr_down = sorted({round(x, 2) for x in closes.nsmallest(6).tolist()})
    return {"sr_up": sr_up, "sr_down": sr_down}

def detect_retest(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    depth = abs((last['close'] - last['ema20']) / last['close'] * 100)
    tag = "touch" if abs(last['close'] - last['ema20'])/last['close'] < 0.003 else \
          ("near_ema20" if depth < 1.5 else "above_ema20")
    vol_con = (df['volume'].iloc[-5:].mean() < df['vol_sma20'].iloc[-1])
    return {"depth_pct": round(depth, 2), "to_ma_tag": tag, "vol_contraction": bool(vol_con)}

def detect_divergence(df: pd.DataFrame) -> Dict[str, str]:
    price = df['close'].tail(30)
    rsi = df['rsi14'].tail(30)
    if price.iloc[-1] >= price.max() - 1e-9 and rsi.iloc[-1] < rsi.max() - 1e-9:
        return {"rsi_price": "bearish"}
    return {"rsi_price": "none"}

# -------------------------
# 3) SR mềm (EMA/SMA/BB) + Volume/Candle confirmations
# -------------------------
def soft_sr_levels(df: pd.DataFrame) -> Dict[str, list]:
    last = df.iloc[-1]
    near_up, near_dn = [], []
    candidates = {
        "BB.upper": float(last['bb_upper']),
        "BB.mid":   float(last['bb_mid']),
        "BB.lower": float(last['bb_lower']),
        "EMA20":    float(last['ema20']),
        "EMA50":    float(last['ema50']),
        "SMA20":    float(last.get('sma20', last['ema20'])),
        "SMA50":    float(last.get('sma50', last['ema50'])),
    }
    px = float(last['close'])
    for name, lvl in candidates.items():
        if np.isnan(lvl): 
            continue
        if lvl > px: near_up.append((name, lvl))
        if lvl < px: near_dn.append((name, lvl))
    near_up  = [dict(name=n, level=l) for n,l in sorted(near_up, key=lambda x: x[1])]
    near_dn  = [dict(name=n, level=l) for n,l in sorted(near_dn, key=lambda x: x[1], reverse=True)]
    return {"soft_up": near_up, "soft_down": near_dn}

def volume_confirmations(df: pd.DataFrame) -> Dict[str, Any]:
    vr = float(df['vol_ratio'].iloc[-1]) if 'vol_ratio' in df.columns else 1.0
    vz = float(df['vol_z20'].iloc[-1]) if 'vol_z20' in df.columns else 0.0
    vol_contraction = df['volume'].tail(3).mean() < df['vol_sma20'].iloc[-1]
    v5 = df['volume'].tail(5).mean(); v10 = df['volume'].tail(10).mean()
    pb_healthy = vol_contraction and (v5 < v10)
    return {
        "vol_ratio": vr,
        "vol_z20": vz,
        "breakout_vol_ok": (vr >= 1.5) or (vz >= 1.0),
        "breakdown_vol_ok": (vr >= 1.5) or (vz >= 1.0),
        "pullback_vol_healthy": bool(pb_healthy),
    }

def candle_flags(df: pd.DataFrame) -> Dict[str, bool]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    body = float(last.get('body_pct', 0.0))
    uw   = float(last.get('upper_wick_pct', 0.0))
    lw   = float(last.get('lower_wick_pct', 0.0))
    green = last['close'] > last['open']; red = last['close'] < last['open']
    bullish_pin = (lw >= 50) and (body <= 30) and green
    bearish_pin = (uw >= 50) and (body <= 30) and red
    bull_engulf = (green and prev['close'] < prev['open'] and
                   last['close'] > prev['open'] and last['open'] < prev['close'])
    bear_engulf = (red and prev['close'] > prev['open'] and
                   last['close'] < prev['open'] and last['open'] > prev['close'])
    inside = (last['high'] <= prev['high']) and (last['low'] >= prev['low'])
    return {
        "bullish_pin": bool(bullish_pin),
        "bearish_pin": bool(bearish_pin),
        "bullish_engulf": bool(bull_engulf),
        "bearish_engulf": bool(bear_engulf),
        "inside_bar": bool(inside),
    }

# -------------------------
# 4) Breakout helpers
# -------------------------
def recent_swing_high(swings: List[Dict[str, Any]]) -> Optional[float]:
    for s in reversed(swings):
        if s.get("type") == "HH":
            return float(s["price"])
    return None

def detect_breakout(df: pd.DataFrame, swings: List[Dict[str, Any]], vol_thr: float = 1.5) -> dict:
    levels = []
    hh = recent_swing_high(swings)
    confirmed = False
    if hh is not None:
        levels.append(hh)
        close = float(df["close"].iloc[-1])
        vol_ratio = float(df["vol_ratio"].iloc[-1]) if "vol_ratio" in df.columns else 1.0
        vol_z = float(df.get("vol_z20", pd.Series([0])).iloc[-1]) if "vol_z20" in df.columns else 0.0
        vol_ok = (vol_ratio >= vol_thr) or (vol_z >= 1.0)
        confirmed = (close > hh) and vol_ok
    return {"breakout_levels": levels, "last_breakout_confirmed": confirmed}

# -------------------------
# 5) Cluster SR thành TP bands + ETA
# -------------------------
def cluster_levels(levels: List[float], atr: float, k: float = 0.7):
    if atr is None or atr <= 0 or not levels:
        return []
    levels = sorted(levels)
    bands = []
    for p in levels:
        if not bands or abs(p - bands[-1][-1]) > k * atr:
            bands.append([p])
        else:
            bands[-1].append(p)
    out = []
    for grp in bands:
        lo, hi = min(grp), max(grp)
        tp = round((lo + hi) / 2, 2)
        out.append({"band": [lo, hi], "tp": tp})
    return out

def _tf_to_hours(tf: str) -> int:
    tf = tf.upper()
    if tf.endswith("H"): return int(tf[:-1])
    if tf.endswith("D"): return int(tf[:-1]) * 24
    if tf.endswith("W"): return int(tf[:-1]) * 24 * 7
    return 24

def eta_for_bands(close: float, bands, atr: float, tf_hours: int, coef: float = 1.0):
    outs = []
    atr = max(atr, 1e-9)
    for b in bands:
        tp = float(b["tp"])
        bars = int(np.ceil(abs(tp - close) / atr * coef))
        bars = max(bars, 1)
        hours = bars * tf_hours
        outs.append({
            "band": b["band"],
            "tp": tp,
            "eta_bars": bars,
            "eta_hours": hours,
            "eta_days": round(hours / 24, 2)
        })
    return outs

# -------------------------
# 6) Build STRUCT JSON (có context/liquidity/futures sentiment)
# -------------------------
def build_struct_json(
    symbol: str,
    tf: str,
    df: pd.DataFrame,
    context_df: Optional[pd.DataFrame] = None,
    liquidity_zones: Optional[List[Dict[str, Any]]] = None,
    futures_sentiment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    swings = find_swings(df)
    trend = detect_trend(df, swings)
    sr = find_sr(df, swings)
    pullback = detect_retest(df)
    div = detect_divergence(df)
    bo = detect_breakout(df, swings, vol_thr=1.5)

    # Flags BB
    flags = {
        "riding_upper": bool(df['close'].iloc[-1] > df['bb_mid'].iloc[-1]),
        "bb_squeeze": bool(df['bb_width_pct'].iloc[-1] < df['bb_width_pct'].tail(50).median())
    }

    close = float(df['close'].iloc[-1])
    atr = float(df['atr14'].iloc[-1] or 0.0)
    tf_hours = _tf_to_hours(tf)

    # Bands + ETA
    bands = cluster_levels(sr.get('sr_up', [])[:6], atr=atr, k=0.7)
    coef = 1.0
    if flags["riding_upper"]: coef *= 0.7
    if flags["bb_squeeze"]:   coef *= 1.3
    eta_bands = eta_for_bands(close, bands, atr, tf_hours, coef)

    volc = volume_confirmations(df)
    soft = soft_sr_levels(df)
    cndl = candle_flags(df)
    ms_tags = classify_market_structure_from_swings(swings)

    struct: Dict[str, Any] = {
        "symbol": symbol,
        "asof": str(df.index[-1]),
        "timeframe": tf,
        "snapshot": {
            "price": {"open": float(df['open'].iloc[-1]),
                      "high": float(df['high'].iloc[-1]),
                      "low":  float(df['low'].iloc[-1]),
                      "close": close},
            "ma": {"ema20": float(df['ema20'].iloc[-1]), "ema50": float(df['ema50'].iloc[-1])},
            "bb": {"upper": float(df['bb_upper'].iloc[-1]),
                   "mid":   float(df['bb_mid'].iloc[-1]),
                   "lower": float(df['bb_lower'].iloc[-1]),
                   "width_pct": float(df['bb_width_pct'].iloc[-1])},
            "rsi14": float(df['rsi14'].iloc[-1]),
            "atr14": atr,
            "volume": {"last": float(df['volume'].iloc[-1]),
                       "sma20": float(df['vol_sma20'].iloc[-1])},
        },
        "structure": {
            "swings": swings,
            "trend": trend,
            "pullback": {**pullback, "vol_healthy": volc["pullback_vol_healthy"]},
            "bb_flags": {
                "riding_upper_band": flags["riding_upper"],
                "bb_contraction": flags["bb_squeeze"]
            },
            "market_structure": ms_tags,
        },
        "events": {**bo, "breakout_vol_ok": volc["breakout_vol_ok"]},
        "divergence": div,
        "levels": {**sr, "soft_sr": soft},     # SR cứng + SR mềm
        "targets": {"up_bands": bands},
        "eta_hint": {"method": "ATR", "per": "bar", "up_bands": eta_bands},
        "confirmations": {"volume": volc, "candles": cndl},
    }

    # Optional: multi-timeframe context
    if context_df is not None:
        ctx_sw = find_swings(context_df)
        ctx_trend = detect_trend(context_df, ctx_sw)
        ctx_sr = find_sr(context_df, ctx_sw)
        ctx_soft = soft_sr_levels(context_df)

        # Lưu vào struct
        struct["context_trend"] = ctx_trend
        struct["context_levels"] = {**ctx_sr, "soft_sr": ctx_soft}

        # Đánh giá alignment giữa TF hiện tại và TF lớn
        align = None
        if trend["state"] in ("up", "down") and ctx_trend["state"] in ("up", "down"):
            align = (trend["state"] == ctx_trend["state"])

        # Nearest SR ở TF lớn so với giá hiện tại
        ctx_next_up = [lvl for lvl in sorted(ctx_sr.get("sr_up", [])) if lvl > close]
        ctx_next_dn = [lvl for lvl in sorted(ctx_sr.get("sr_down", []), reverse=True) if lvl < close]

        struct["context_guidance"] = {
            "trend_aligned": bool(align) if align is not None else None,
            "nearest_resistance": ctx_next_up[0] if ctx_next_up else None,
            "nearest_support": ctx_next_dn[0] if ctx_next_dn else None,
        }

    # Optional: liquidity zones (truyền từ ngoài để tái sử dụng/tùy biến tham số)
    if liquidity_zones is not None:
        struct["liquidity_zones"] = liquidity_zones

    # Futures sentiment: nếu caller không truyền, tự gọi indicators.fetch_funding_oi(symbol)
    if futures_sentiment is not None:
        struct["futures_sentiment"] = futures_sentiment
    else:
        try:
            from indicators import fetch_funding_oi
            struct["futures_sentiment"] = fetch_funding_oi(symbol)
        except Exception as e:
            struct["futures_sentiment"] = {"error": str(e)}

    return struct

    
