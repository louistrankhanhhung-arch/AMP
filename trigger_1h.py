from typing import Dict, Any, Tuple


# -------------------------
# Helpers
# -------------------------
def _get(d, *keys, default=None):
    """Lấy giá trị lồng nhau trong dict, nếu không có thì trả về default."""
    cur = d or {}
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _near(val, ref, atr, k=0.25) -> bool:
    """Kiểm tra giá trị val có gần ref trong phạm vi k * ATR hay không."""
    try:
        return abs(float(val) - float(ref)) <= (k * float(atr))
    except Exception:
        return False


def _candles_any(candles: Dict[str, bool], keys) -> bool:
    """Kiểm tra xem candles có chứa ít nhất một mẫu nến trong keys hay không."""
    if not isinstance(candles, dict):
        return False
    return any(bool(candles.get(k)) for k in keys)


def _common_vals(s: Dict) -> Tuple[float, float, float, float]:
    """Trích xuất giá close, EMA20, RSI14, ATR14 từ snapshot."""
    close = float(_get(s, "snapshot", "price", "close", default=0.0) or 0.0)
    ema20 = float(_get(s, "snapshot", "ema20", default=0.0) or 0.0)
    rsi14 = float(_get(s, "snapshot", "rsi14", default=50.0) or 50.0)
    atr14 = float(_get(s, "snapshot", "atr14", default=0.0) or 0.0)
    return close, ema20, rsi14, atr14


# -------------------------
# Long Trigger
# -------------------------
def check_long_trigger(s1h: Dict) -> Dict[str, Any]:
    close, ema20, rsi14, atr = _common_vals(s1h)
    events = _get(s1h, "events", default={}) or {}
    pb = _get(s1h, "structure", "pullback", default={}) or {}
    vol = _get(s1h, "confirmations", "volume", default={}) or {}
    candles = _get(s1h, "confirmations", "candles", default={}) or {}

    # Breakout xác nhận
    if bool(events.get("last_breakout_confirmed")):
        return {"ok": True, "type": "breakout", "reasons": ["last_breakout_confirmed=1"]}

    # Reclaim EMA20
    reclaim = (
        pb.get("to_ma_tag") in ("touch", "near_ema20")
        and (close >= ema20)
        and (rsi14 > 50)
        and bool(vol.get("pullback_vol_healthy"))
    )
    if reclaim:
        return {
            "ok": True,
            "type": "reclaim",
            "reasons": ["to_ma_tag_ok", "close>=ema20", "rsi>50", "pullback_vol_healthy"],
        }

    # Retest EMA20
    rej_candle = _candles_any(candles, ["bullish_engulfing", "pin_bar_bull"])
    retest = rej_candle and _near(close, ema20, atr, k=0.25) and bool(vol.get("pullback_vol_healthy"))
    if retest:
        return {
            "ok": True,
            "type": "retest",
            "reasons": ["bullish_reject", "near_ema20", "pullback_vol_healthy"],
        }

    return {"ok": False, "type": None, "reasons": []}


# -------------------------
# Short Trigger
# -------------------------
def check_short_trigger(s1h: Dict) -> Dict[str, Any]:
    close, ema20, rsi14, atr = _common_vals(s1h)
    events = _get(s1h, "events", default={}) or {}
    pb = _get(s1h, "structure", "pullback", default={}) or {}
    vol = _get(s1h, "confirmations", "volume", default={}) or {}
    candles = _get(s1h, "confirmations", "candles", default={}) or {}

    # Breakdown xác nhận
    if bool(events.get("last_breakdown_confirmed")):
        return {"ok": True, "type": "breakdown", "reasons": ["last_breakdown_confirmed=1"]}

    # Reclaim xuống EMA20
    reclaim_down = (
        pb.get("to_ma_tag") in ("touch", "near_ema20")
        and (close <= ema20)
        and (rsi14 < 50)
        and bool(vol.get("pullback_vol_healthy"))
    )
    if reclaim_down:
        return {
            "ok": True,
            "type": "reclaim_down",
            "reasons": ["to_ma_tag_ok", "close<=ema20", "rsi<50", "pullback_vol_healthy"],
        }

    # Retest EMA20 xuống
    rej_candle = _candles_any(candles, ["bearish_engulfing", "pin_bar_bear"])
    retest = rej_candle and _near(close, ema20, atr, k=0.25) and bool(vol.get("pullback_vol_healthy"))
    if retest:
        return {
            "ok": True,
            "type": "retest_down",
            "reasons": ["bearish_reject", "near_ema20", "pullback_vol_healthy"],
        }

    return {"ok": False, "type": None, "reasons": []}
