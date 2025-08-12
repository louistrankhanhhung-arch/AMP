# structure_engine.py
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np

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

# -------------------------
# 2) Trend / SR / Pullback / Divergence / Volume / Candle
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

# structure_engine.py — thêm helpers

def soft_sr_levels(df: pd.DataFrame, side: str = "long") -> Dict[str, list]:
    """Sinh SR mềm từ EMA/SMA/BB."""
    last = df.iloc[-1]
    near_up = []
    near_dn = []
    # Các mức tiềm năng
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

    # contraction check: trung bình 3 nến gần nhất < SMA20
    vol_contraction = df['volume'].tail(3).mean() < df['vol_sma20'].iloc[-1]
    # xu hướng vol giảm trong pullback: 3 nến liên tiếp giảm hoặc trung bình 5 nến < 10 nến
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
    """Một số mẫu nến & cụm nến hay dùng."""
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    body = float(last['body_pct']) if 'body_pct' in df.columns else 0.0
    uw   = float(last['upper_wick_pct']) if 'upper_wick_pct' in df.columns else 0.0
    lw   = float(last['lower_wick_pct']) if 'lower_wick_pct' in df.columns else 0.0
    green = last['close'] > last['open']
    red   = last['close'] < last['open']

    # Pin bar / Hammer / Shooting star
    bullish_pin = (lw >= 50) and (body <= 30) and green
    bearish_pin = (uw >= 50) and (body <= 30) and red

    # Engulfing
    bull_engulf = (green and prev['close'] < prev['open'] and
                   last['close'] > prev['open'] and last['open'] < prev['close'])
    bear_engulf = (red and prev['close'] > prev['open'] and
                   last['close'] < prev['open'] and last['open'] > prev['close'])

    # Inside bar breakout (đơn giản)
    inside = (last['high'] <= prev['high']) and (last['low'] >= prev['low'])
    breakout_up_next = False
    breakout_dn_next = False

    return {
        "bullish_pin": bool(bullish_pin),
        "bearish_pin": bool(bearish_pin),
        "bullish_engulf": bool(bull_engulf),
        "bearish_engulf": bool(bear_engulf),
        "inside_bar": bool(inside),
        "breakout_up_next": bool(breakout_up_next),
        "breakout_dn_next": bool(breakout_dn_next),
    }

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

# --- Breakout helpers (dùng List[Dict[str,Any]] để tương thích tốt) ---
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
        vol_ok = vol_ratio >= vol_thr or float(df.get("vol_z20", pd.Series([0])).iloc[-1]) >= 1.0
        confirmed = (close > hh) and vol_ok
    return {"breakout_levels": levels, "last_breakout_confirmed": confirmed}

# -------------------------
# 3) Cluster SR thành TP bands + ETA
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
# 4) Build STRUCT JSON
# -------------------------
def build_struct_json(symbol: str, tf: str, df: pd.DataFrame) -> Dict[str, Any]:
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
    
    struct = {
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
        "pullback": {
            **pullback,
            "vol_healthy": volc["pullback_vol_healthy"]  # xác nhận pullback lành mạnh
        },
        "bb_flags": {
            "riding_upper_band": flags["riding_upper"],
            "bb_contraction": flags["bb_squeeze"]
        }
    },
    "events": {**bo, "breakout_vol_ok": volc["breakout_vol_ok"]},
    "divergence": div,
    "levels": {**sr, "soft_sr": soft},      # GIỮ SR gốc & thêm SR mềm
    "targets": {"up_bands": bands},         # TP bands (hard SR gom)
    "eta_hint": {"method": "ATR", "per": "bar", "up_bands": eta_bands},
    "confirmations": {
        "volume": volc,
        "candles": cndl
    }
}
    return struct
