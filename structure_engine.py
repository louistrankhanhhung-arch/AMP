# structure_engine.py
from typing import List, Dict, Any, Tuple
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
        if last_ext == 0:
            change_pct = 0
        else:
            change_pct = (v - last_ext) / last_ext * 100
        if direction >= 0 and change_pct >= pct:
            pts.append((last_t, last_ext))
            last_ext, last_t, direction = v, t, 1
        elif direction <= 0 and change_pct <= -pct:
            pts.append((last_t, last_ext))
            last_ext, last_t, direction = v, t, -1
        else:
            # update extreme in the current direction
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
    return out[-20:]  # giữ 20 điểm gần nhất

# -------------------------
# 2) Trend / SR / Pullback / Divergence
# -------------------------
def detect_trend(df: pd.DataFrame, swings) -> Dict[str, Any]:
    ema20, ema50 = df['ema20'].iloc[-1], df['ema50'].iloc[-1]
    state = "up" if ema20 > ema50 else ("down" if ema20 < ema50 else "side")
    age = min(len(df), 100)
    return {"state": state, "basis": "ema20 vs ema50", "age_bars": age}

def find_sr(df: pd.DataFrame, swings) -> Dict[str, list]:
    # đơn giản: lấy vài đỉnh/đáy gần nhất làm SR
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
# (1) Thêm lại vào structure_engine.py, ngay dưới detect_divergence()

def recent_swing_high(swings: list[dict]) -> float | None:
    """Lấy đỉnh (HH) gần nhất trong list swings, nếu có."""
    for s in reversed(swings):
        if s.get("type") == "HH":
            return float(s["price"])
    return None

def detect_breakout(df: pd.DataFrame, swings: list[dict], vol_thr: float = 1.5) -> dict:
    """
    Breakout đơn giản:
    - Close hiện tại > swing-high gần nhất (nếu có), và
    - Khối lượng hiện tại >= vol_thr * SMA20 (dùng vol_ratio)
    Trả về: {"levels":[...], "last_breakout_confirmed": bool}
    """
    levels = []
    hh = recent_swing_high(swings)
    confirmed = False
    if hh is not None:
        levels.append(hh)
        close = float(df["close"].iloc[-1])
        vol_ratio = float(df["vol_ratio"].iloc[-1]) if "vol_ratio" in df.columns else 1.0
        confirmed = (close > hh) and (vol_ratio >= vol_thr)
    return {"breakout_levels": levels, "last_breakout_confirmed": confirmed}

# -------------------------
# 3) Cluster SR thành TP bands + ETA
# -------------------------
def cluster_levels(levels: List[float], atr: float, k: float = 0.7):
    """
    Gom các mức SR gần nhau thành band nếu khoảng cách < k * ATR.
    Trả về list dict: {"band":[low, high], "tp": midpoint}
    """
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
    if tf.endswith("H"):
        return int(tf[:-1])
    if tf.endswith("D"):
        return int(tf[:-1]) * 24
    if tf.endswith("W"):
        return int(tf[:-1]) * 24 * 7
    return 24  # default

def eta_for_bands(close: float, bands, atr: float, tf_hours: int, coef: float = 1.0):
    """
    Tính ETA cho từng band theo ATR, trả về list:
    {"band":[lo,hi], "tp":x, "eta_bars":n, "eta_hours":h, "eta_days":d}
    """
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
# 4) Build STRUCT JSON (giữ SR gốc + thêm bands & ETA)
# -------------------------
def build_struct_json(symbol: str, tf: str, df: pd.DataFrame) -> Dict[str, Any]:
    swings = find_swings(df)
    trend = detect_trend(df, swings)
    sr = find_sr(df, swings)
    pullback = detect_retest(df)
    div = detect_divergence(df)
    bo = detect_breakout(df, swings, vol_thr=1.5)

    # flags ngữ cảnh cho ETA
    flags = {
        "riding_upper": bool((df['close'].iloc[-1] > df['bb_mid'].iloc[-1])),
        "bb_squeeze": bool(df['bb_width_pct'].iloc[-1] < df['bb_width_pct'].tail(50).median())
    }

    close = float(df['close'].iloc[-1])
    atr = float(df['atr14'].iloc[-1] or 0.0)
    tf_hours = _tf_to_hours(tf)

    # --- NEW: gom SR_up thành bands & tính ETA theo bars/hours/days ---
    bands = cluster_levels(sr.get('sr_up', [])[:6], atr=atr, k=0.7)  # bạn có thể chỉnh k
    # hệ số ngữ cảnh
    coef = 1.0
    if flags["riding_upper"]:
        coef *= 0.7
    if flags["bb_squeeze"]:
        coef *= 1.3
    eta_bands = eta_for_bands(close, bands, atr, tf_hours, coef)

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
            "pullback": pullback,
            "bb_behaviour": "riding_upper_band" if flags["riding_upper"] else "below_mid"
            "events": bo,
        },
        "events": {"breakout_levels": [], "last_breakout_confirmed": False},
        "divergence": div,

        # GIỮ SR gốc để GPT tham chiếu khi cần
        "levels": sr,

        # NEW: targets theo bands (đã gom) + ETA chi tiết
        "targets": {"up_bands": bands},
        "eta_hint": {"method": "ATR", "per": "bar", "up_bands": eta_bands}
    }
    return struct
