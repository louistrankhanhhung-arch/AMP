from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

def _zigzag(series: pd.Series, pct: float = 2.0) -> List[Tuple[pd.Timestamp, float]]:
    points = []
    if series.empty:
        return points
    last_ext = series.iloc[0]
    last_t = series.index[0]
    direction = 0  # 1 up, -1 down, 0 none
    for t, v in series.items():
        change_pct = (v - last_ext) / last_ext * 100 if last_ext != 0 else 0
        if direction >= 0 and change_pct >= pct:
            points.append((last_t, last_ext))
            last_ext, last_t, direction = v, t, 1
        elif direction <= 0 and change_pct <= -pct:
            points.append((last_t, last_ext))
            last_ext, last_t, direction = v, t, -1
        else:
            if (direction >= 0 and v > last_ext) or (direction <= 0 and v < last_ext):
                last_ext, last_t = v, t
    points.append((last_t, last_ext))
    return points

def find_swings(df: pd.DataFrame, zigzag_pct: float = 2.0):
    zz = _zigzag(df['close'], zigzag_pct)
    out = []
    for i in range(1, len(zz)):
        prev, curr = zz[i-1][1], zz[i][1]
        t = zz[i][0]
        if curr > prev: out.append({"type":"HH","t":str(t), "price":float(curr)})
        else: out.append({"type":"LL","t":str(t), "price":float(curr)})
    # post-process to mark HL/LH roughly
    return out[-20:]  # keep last N

def detect_trend(df: pd.DataFrame, swings) -> Dict[str, Any]:
    ema20, ema50 = df['ema20'].iloc[-1], df['ema50'].iloc[-1]
    state = "up" if ema20 > ema50 else ("down" if ema20 < ema50 else "side")
    age = min(len(df), 100)
    return {"state": state, "basis": "ema20 vs ema50", "age_bars": age}

def find_sr(df: pd.DataFrame, swings) -> Dict[str, list]:
    closes = df['close']
    up = sorted({round(x,2) for x in closes.tail(200).nlargest(5).tolist()})
    down = sorted({round(x,2) for x in closes.tail(200).nsmallest(5).tolist()})
    return {"sr_up": up, "sr_down": down}

def detect_retest(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    depth = abs((last['close'] - last['ema20']) / last['close'] * 100)
    tag = "touch" if abs(last['close'] - last['ema20'])/last['close'] < 0.003 else ("near_ema20" if depth < 1.5 else "above_ema20")
    vol_con = (df['volume'].iloc[-5:].mean() < df['vol_sma20'].iloc[-1])
    return {"depth_pct": round(depth,2), "to_ma_tag": tag, "vol_contraction": bool(vol_con)}

def detect_breakout(df: pd.DataFrame, level: float) -> bool:
    recent = df['close'].iloc[-3:]
    vol = df['vol_ratio'].iloc[-1]
    return bool((recent.max() > level) and (vol >= 1.5))

def detect_divergence(df: pd.DataFrame) -> Dict[str, str]:
    # Simple heuristic: higher high in price but lower high in RSI over last 30 bars
    price = df['close'].tail(30)
    rsi = df['rsi14'].tail(30)
    if price.iloc[-1] > price.max()-1e-9 and rsi.iloc[-1] < rsi.max()-1e-9:
        return {"rsi_price":"bearish"}
    return {"rsi_price":"none"}

def estimate_eta(close: float, targets: List[float], atr14: float, flags: Dict[str, Any]) -> List[int]:
    coef = 1.0
    if flags.get("riding_upper", False): coef *= 0.7
    if flags.get("bb_squeeze", False): coef *= 1.3
    if flags.get("thick_sr", False): coef *= 1.2
    def _eta(tp): 
        dist = abs(tp - close)
        days = int(np.ceil((dist / max(atr14, 1e-9)) * coef))
        return max(days, 1)
    return [_eta(tp) for tp in targets]

def build_struct_json(symbol: str, tf: str, df: pd.DataFrame) -> Dict[str, Any]:
    swings = find_swings(df)
    trend = detect_trend(df, swings)
    sr = find_sr(df, swings)
    pullback = detect_retest(df)
    div = detect_divergence(df)
    flags = {"riding_upper": bool((df['close'].iloc[-1] > df['bb_mid'].iloc[-1])),
             "bb_squeeze": bool(df['bb_width_pct'].iloc[-1] < df['bb_width_pct'].tail(50).median())}
    targets = sr['sr_up'][:3]
    eta = estimate_eta(float(df['close'].iloc[-1]), targets, float(df['atr14'].iloc[-1]), flags)
    struct = {
        "symbol": symbol,
        "asof": str(df.index[-1]),
        "timeframe": tf,
        "snapshot": {
            "price": {
                "open": float(df['open'].iloc[-1]),
                "high": float(df['high'].iloc[-1]),
                "low": float(df['low'].iloc[-1]),
                "close": float(df['close'].iloc[-1])
            },
            "ma": {"ema20": float(df['ema20'].iloc[-1]), "ema50": float(df['ema50'].iloc[-1])},
            "bb": {"upper": float(df['bb_upper'].iloc[-1]), "mid": float(df['bb_mid'].iloc[-1]), "lower": float(df['bb_lower'].iloc[-1]), "width_pct": float(df['bb_width_pct'].iloc[-1])},
            "rsi14": float(df['rsi14'].iloc[-1]),
            "atr14": float(df['atr14'].iloc[-1]),
            "volume": {"last": float(df['volume'].iloc[-1]), "sma20": float(df['vol_sma20'].iloc[-1])},
        },
        "structure": {
            "swings": swings,
            "trend": trend,
            "pullback": pullback,
            "bb_behaviour": "riding_upper_band" if flags["riding_upper"] else "below_mid"
        },
        "events": {"breakout_levels": [], "last_breakout_confirmed": False},
        "divergence": div,
        "levels": sr,
        "eta_hint": {"method":"ATR","est_days": eta}
    }
    return struct
