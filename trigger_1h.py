# trigger_1h.py
from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime

# Dùng các module sẵn có của bạn
from kucoin_api import fetch_batch
from indicators import enrich_indicators, enrich_more
from structure_engine import build_struct_json

def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _drop_incomplete_last(df, tf_seconds: int) -> Any:
    """
    Loại bỏ nến đang chạy ở cuối (nếu có) theo tf_seconds.
    Giả định df['timestamp'] là epoch ms hoặc s (cứ //1000 về giây).
    """
    if df is None or len(df) == 0:
        return df
    import time
    last = int(df["timestamp"].iloc[-1] // 1000)
    now  = int(time.time())
    if now < last + tf_seconds:
        return df.iloc[:-1].copy()
    return df

def _bool(x) -> Optional[bool]:
    return None if x is None else bool(x)

def check_long_trigger(struct_1h: Dict[str, Any]) -> bool:
    ev   = (struct_1h.get("events") or {})
    snap = (struct_1h.get("snapshot") or {})
    ok_breakout = bool(ev.get("last_breakout_confirmed", False))
    close = snap.get("close"); ema20 = snap.get("ema20"); rsi = snap.get("rsi")
    momentum_ok = (close is not None and ema20 is not None and close > ema20)
    rsi_ok = (rsi is None) or (rsi >= 50)
    return bool(ok_breakout and momentum_ok and rsi_ok)

def check_short_trigger(struct_1h: Dict[str, Any]) -> bool:
    ev   = (struct_1h.get("events") or {})
    snap = (struct_1h.get("snapshot") or {})
    ok_breakdown = bool(ev.get("last_breakdown_confirmed", False))  # <-- dùng khóa mới từ engine
    close = snap.get("close"); ema20 = snap.get("ema20"); rsi = snap.get("rsi")
    momentum_ok = (close is not None and ema20 is not None and close < ema20)
    rsi_ok = (rsi is None) or (rsi <= 50)
    return bool(ok_breakdown and momentum_ok and rsi_ok)

def build_trigger_for_symbol(symbol: str, limit: int = 180) -> Dict[str, Any]:
    """
    Fetch 1H cho 1 mã, dùng NẾN ĐÃ ĐÓNG, build struct 1H, và suy ra trigger.
    Trả về dict đơn giản để feed GPT (hoặc dùng trong scheduler).
    """
    # 1) fetch & ensure closed bars
    batch = fetch_batch(symbol, timeframes=["1H"], limit=limit)
    df = batch.get("1H")
    if df is None or len(df) == 0:
        return {"symbol": symbol, "timeframe": "1H", "ok": False, "error": "no_ohlcv"}

    df = _drop_incomplete_last(df, tf_seconds=3600)
    if df is None or len(df) == 0:
        return {"symbol": symbol, "timeframe": "1H", "ok": False, "error": "no_closed_bar"}

    # 2) enrich indicators
    df = enrich_more(enrich_indicators(df))

    # 3) build struct 1H bằng engine (engine đã dùng nến đóng)
    s1h = build_struct_json(symbol, "1H", df)

    # 4) derive triggers
    long_trig  = check_long_trigger(s1h)
    short_trig = check_short_trigger(s1h)

    # 5) Một số flag gợi ý thêm
    snap = s1h.get("snapshot", {})
    trig = {
        "symbol": symbol,
        "timeframe": "1H",
        "ok": True,
        "reclaim_ma20": _bool(snap.get("close") is not None and snap.get("ema20") is not None and snap["close"] > snap["ema20"]),
        "rsi_gt_50": _bool((snap.get("rsi") or 0) > 50),
        "last_breakout_confirmed": bool((s1h.get("events") or {}).get("last_breakout_confirmed", False)),
        "last_breakdown_confirmed": bool((s1h.get("events") or {}).get("last_breakdown_confirmed", False)),
        "long_trigger": bool(long_trig),
        "short_trigger": bool(short_trig),
        "note": "trigger-1H based on closed bars",
    }
    return trig
