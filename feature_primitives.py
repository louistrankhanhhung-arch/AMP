"""
feature_primitives.py
---------------------
Feature primitives tách rời để dễ debug & tái sử dụng.

• Cụm Xu Hướng (trend)
    - compute_swings(df, pct=2.0, max_keep=20, last_n_each=3)
    - compute_trend(df)
    - compute_candles(df)

• Cụm Động Lượng (momentum)
    - compute_volume_features(df)
    - compute_momentum(df)
    - compute_volatility(df, bbw_lookback=50)

• Cụm SR (support/resistance)
    - compute_levels(df, atr=None, tol_coef=0.5, extremes=12, lookback=300)
    - compute_soft_levels(df)

• Tổng hợp đa khung thời gian
    - compute_features_by_tf(dfs_by_tf: Dict[str, pd.DataFrame]) -> Dict

YÊU CẦU CỘT (đã enrich trước):
open, high, low, close, volume, ema20, ema50, rsi14, atr14,
bb_upper, bb_mid, bb_lower, vol_sma20, vol_ratio, vol_z20,
(body_pct, upper_wick_pct, lower_wick_pct) nếu muốn mẫu nến chuẩn.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

# =============================
# Helpers chung
# =============================

def _last_closed_bar(df: pd.DataFrame) -> pd.Series:
    """Trả về nến đã đóng cuối cùng nếu có >=2 nến, ngược lại lấy nến cuối.
    Dùng cho pattern candle để an toàn trong streaming.
    """
    if df is None or len(df) == 0:
        raise ValueError("empty df")
    if len(df) >= 2:
        return df.iloc[-2]
    return df.iloc[-1]


# =============================
# CỤM XU HƯỚNG
# =============================

def _zigzag(series: pd.Series, pct: float = 2.0) -> List[Tuple[pd.Timestamp, float]]:
    """ZigZag đơn giản theo % (trên chuỗi close). Trả [(t, price), ...]."""
    pts: List[Tuple[pd.Timestamp, float]] = []
    if series is None or series.empty:
        return pts

    last_ext = float(series.iloc[0])
    last_t = series.index[0]
    direction = 0  # 1 up, -1 down, 0 none

    for t, v in series.items():
        v = float(v)
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


def compute_swings(
    df: pd.DataFrame,
    pct: float = 2.0,
    *,
    lookback: int = 250,
    max_keep: int = 20,
    last_n_each: int = 3,
) -> Dict[str, Any]:
    """Tạo danh sách HH/LL từ ZigZag.

    Args:
        pct: ngưỡng ZigZag theo %.
        lookback: số nến lấy để tính.
        max_keep: giới hạn tổng số swing lưu lại để nhẹ payload.
        last_n_each: số lượng HH/LL gần nhất muốn trích riêng (ví dụ 3HH-3LL như bạn đề xuất).
    Returns:
        {
          'swings': [{'type': 'HH'|'LL', 't': str, 'price': float}, ...],
          'last_HH': [float,...],
          'last_LL': [float,...]
        }
    """
    series = df['close'].tail(lookback)
    zz = _zigzag(series, pct=pct)
    out: List[Dict[str, Any]] = []
    for i in range(1, len(zz)):
        prev, curr = zz[i - 1][1], zz[i][1]
        t = zz[i][0]
        out.append({
            "type": "HH" if curr > prev else "LL",
            "t": str(t),
            "price": float(curr)
        })
    swings = out[-max_keep:]

    # Trích 3HH/3LL (mặc định)
    last_HH = [s['price'] for s in reversed(swings) if s['type'] == 'HH'][:last_n_each]
    last_LL = [s['price'] for s in reversed(swings) if s['type'] == 'LL'][:last_n_each]

    return {"swings": swings, "last_HH": last_HH, "last_LL": last_LL}


def compute_trend(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.sort_index()
    e20, e50 = float(df['ema20'].iloc[-1]), float(df['ema50'].iloc[-1])
    spread = e20 - e50
    if e20 > e50:
        state = "up"
    elif e20 < e50:
        state = "down"
    else:
        state = "side"
    return {"state": state, "ema20": e20, "ema50": e50, "ema_spread": spread}


def compute_candles(df: pd.DataFrame) -> Dict[str, bool]:
    last = _last_closed_bar(df)
    prev = df.iloc[-3] if len(df) >= 3 else last

    body = float(last.get('body_pct', 0.0))
    uw = float(last.get('upper_wick_pct', 0.0))
    lw = float(last.get('lower_wick_pct', 0.0))

    green = bool(last['close'] > last['open'])
    red = bool(last['close'] < last['open'])

    bullish_pin = (lw >= 50) and (body <= 30) and green
    bearish_pin = (uw >= 50) and (body <= 30) and red

    bull_engulf = (
        green
        and prev['close'] < prev['open']
        and last['close'] > prev['open']
        and last['open'] < prev['close']
    )
    bear_engulf = (
        red
        and prev['close'] > prev['open']
        and last['close'] < prev['open']
        and last['open'] > prev['close']
    )

    inside = (last['high'] <= prev['high']) and (last['low'] >= prev['low'])

    return {
        "bullish_pin": bool(bullish_pin),
        "bearish_pin": bool(bearish_pin),
        "bullish_engulf": bool(bull_engulf),
        "bearish_engulf": bool(bear_engulf),
        "inside_bar": bool(inside),
    }


# =============================
# CỤM ĐỘNG LƯỢNG
# =============================

def compute_volume_features(df: pd.DataFrame) -> Dict[str, Any]:
    vr = float(df['vol_ratio'].iloc[-1]) if 'vol_ratio' in df.columns else 1.0
    vz = float(df['vol_z20'].iloc[-1]) if 'vol_z20' in df.columns else 0.0

    v3 = float(df['volume'].tail(3).mean())
    v5 = float(df['volume'].tail(5).mean())
    v10 = float(df['volume'].tail(10).mean())
    v20 = float(df['vol_sma20'].iloc[-1]) if 'vol_sma20' in df.columns else v20 if 'v20' in locals() else v10

    contraction = (v5 < v10) and (v10 < v20)

    return {
        "vol_ratio": vr,
        "vol_z20": vz,
        "v3": v3, "v5": v5, "v10": v10, "v20": v20,
        "contraction": bool(contraction),
        "break_vol_ok": bool((vr >= 1.5) or (vz >= 1.0)),
        "break_vol_strong": bool((vr >= 2.0) or (vz >= 2.0)),
    }


def compute_momentum(df: pd.DataFrame) -> Dict[str, Any]:
    price = df['close'].tail(30)
    rsi = df['rsi14'].tail(30) if 'rsi14' in df.columns else pd.Series([50])
    rsi_last = float(rsi.iloc[-1]) if len(rsi) else 50.0

    div = "none"
    if len(price) >= 3 and len(rsi) >= 3:
        if price.iloc[-1] >= price.max() - 1e-9 and rsi.iloc[-1] < rsi.max() - 1e-9:
            div = "bearish"
        elif price.iloc[-1] <= price.min() + 1e-9 and rsi.iloc[-1] > rsi.min() + 1e-9:
            div = "bullish"

    return {"rsi": rsi_last, "divergence": div}


def compute_volatility(df: pd.DataFrame, bbw_lookback: int = 50) -> Dict[str, Any]:
    atr = float(df['atr14'].iloc[-1]) if 'atr14' in df.columns else 0.0

    if 'bb_width_pct' in df.columns and pd.notna(df['bb_width_pct'].iloc[-1]):
        bbw_series = df['bb_width_pct'].copy()
    else:
        upper = df['bb_upper']; lower = df['bb_lower']; mid = df['bb_mid']
        base = mid.where(mid.abs() > 1e-12, other=df['close'])
        bbw_series = ((upper - lower) / base.abs()) * 100.0
        bbw_series = bbw_series.replace([np.inf, -np.inf], np.nan)

    bbw_med = float(bbw_series.tail(bbw_lookback).median(skipna=True)) if len(bbw_series) else 0.0
    bbw_last = float(bbw_series.iloc[-1]) if len(bbw_series) else 0.0
    squeeze = bool(bbw_last < bbw_med) if bbw_med > 0 else False

    return {"atr": atr, "bbw_last": bbw_last, "bbw_med": bbw_med, "squeeze": squeeze}


# =============================
# CỤM SR
# =============================

def compute_levels(
    df: pd.DataFrame,
    atr: Optional[float] = None,
    *,
    tol_coef: float = 0.5,
    extremes: int = 12,
    lookback: int = 300,
) -> Dict[str, Any]:
    """SR cứng từ local HL + extreme closes, cluster bởi tol = tol_coef*ATR.
    Trả ra sr_up/sr_down so với close hiện tại và danh sách bands (clusters).
    """
    sub = df.tail(lookback)
    px = float(sub['close'].iloc[-1])

    if atr is None:
        atr = float(sub['atr14'].iloc[-1]) if 'atr14' in sub.columns else 0.0
    tol = max(atr * tol_coef, 1e-6)

    highs = sub['high']
    lows = sub['low']
    loc_high = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    loc_low = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]

    closes = sub['close']
    extreme_up = closes.nlargest(extremes).tolist()
    extreme_dn = closes.nsmallest(extremes).tolist()

    cands: List[float] = []
    cands += [float(x) for x in loc_high.dropna().tolist()]
    cands += [float(x) for x in loc_low.dropna().tolist()]
    cands += [float(x) for x in extreme_up if np.isfinite(x)]
    cands += [float(x) for x in extreme_dn if np.isfinite(x)]

    cands = sorted(set(round(x, 4) for x in cands))

    # cluster 1D theo tol
    merged: List[float] = []
    for p in cands:
        if not merged:
            merged.append(p)
            continue
        if abs(p - merged[-1]) <= tol:
            merged[-1] = (merged[-1] + p) / 2.0
        else:
            merged.append(p)

    sr_up = sorted({x for x in merged if x > px})
    sr_dn = sorted({x for x in merged if x < px})

    # dựng bands (loose grouping)
    def _bands(levels: List[float]) -> List[Dict[str, Any]]:
        bands: List[List[float]] = []
        for p in levels:
            if not bands or abs(p - bands[-1][-1]) > tol:
                bands.append([p])
            else:
                bands[-1].append(p)
        out = []
        for grp in bands:
            lo, hi = min(grp), max(grp)
            tp = round((lo + hi) / 2.0, 2)
            out.append({"band": [lo, hi], "tp": tp})
        return out

    return {
        "sr_up": [round(x, 4) for x in sr_up],
        "sr_down": [round(x, 4) for x in sr_dn],
        "bands_up": _bands(sr_up),
        "bands_down": _bands(sr_dn),
        "tol": tol,
    }


def compute_soft_levels(df: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    last = df.iloc[-1]
    px = float(last['close'])
    candidates = {
        "BB.upper": float(last['bb_upper']),
        "BB.mid": float(last['bb_mid']),
        "BB.lower": float(last['bb_lower']),
        "EMA20": float(last['ema20']),
        "EMA50": float(last['ema50']),
        "SMA20": float(last.get('sma20', last['ema20'])),
        "SMA50": float(last.get('sma50', last['ema50'])),
    }
    up, dn = [], []
    for name, lvl in candidates.items():
        if not np.isfinite(lvl):
            continue
        if lvl > px:
            up.append((name, lvl))
        elif lvl < px:
            dn.append((name, lvl))
    up = [dict(name=n, level=l) for n, l in sorted(up, key=lambda x: x[1])]
    dn = [dict(name=n, level=l) for n, l in sorted(dn, key=lambda x: x[1], reverse=True)]
    return {"soft_up": up, "soft_down": dn}


# =============================
# ĐA KHUNG THỜI GIAN
# =============================

def compute_features_by_tf(dfs_by_tf: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Tính toàn bộ primitives cho nhiều TF. dfs_by_tf ví dụ: {'1H': df1h, '4H': df4h, '1D': df1d}
    Trả dict theo từng TF với cấu trúc đồng nhất.
    """
    out: Dict[str, Any] = {}
    for tf, df in dfs_by_tf.items():
        if df is None or len(df) == 0:
            out[tf] = {"error": "empty df"}
            continue
        df = df.sort_index()
        swings = compute_swings(df)
        trend = compute_trend(df)
        candles = compute_candles(df)
        vol = compute_volume_features(df)
        mom = compute_momentum(df)
        vola = compute_volatility(df)
        sr = compute_levels(df, atr=vola.get('atr', 0.0))
        soft = compute_soft_levels(df)

        out[tf] = {
            "swings": swings,
            "trend": trend,
            "candles": candles,
            "volume": vol,
            "momentum": mom,
            "volatility": vola,
            "levels": sr,
            "soft_levels": soft,
        }
    return out
