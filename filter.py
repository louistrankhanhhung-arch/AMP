from dataclasses import dataclass
from typing import List, Dict
import math

@dataclass
class FilterResult:
    symbol: str
    score: float
    tag: str
    bucket: str
    reasons: list
    hints: dict

def score_symbol(struct: Dict) -> FilterResult:
    snap = struct.get("snapshot", {})
    lvl = struct.get("levels", {})
    trend = struct.get("structure", {}).get("trend", {}).get("state","side")
    rsi = snap.get("rsi14", 50)
    atr = snap.get("atr14", 1)
    close = snap.get("price",{}).get("close", 0)
    sr_up = lvl.get("sr_up", [])
    nearest_up = min(sr_up, key=lambda x: abs(x-close)) if sr_up else None
    dist_up_atr = (nearest_up - close) / atr if (nearest_up and atr) else 0
    div = struct.get("divergence", {}).get("rsi_price", "none")

    flags = {
        "trend_up": 1 if trend=="up" else 0,
        "momentum_pos": 1 if rsi>50 else 0,
        "no_bear_div": 1 if div!="bearish" else 0,
        "sr_clearance": 1 if dist_up_atr>0.7 else 0
    }
    # naive R:R to tp2 using sr_up[1] if exists
    rr_tp2 = 0.0
    if sr_up and len(sr_up)>=2:
        tp2 = sr_up[1]
        sl = close - 1.2*atr
        rr_tp2 = (tp2 - close) / (close - sl) if (close - sl)!=0 else 0
        flags["rr_tp2_ge_2_5"] = 1 if rr_tp2>=2.5 else 0
    else:
        flags["rr_tp2_ge_2_5"] = 0

    # pullback_ok
    pb = struct.get("structure",{}).get("pullback",{})
    pullback_ok = (pb.get("to_ma_tag") in ["touch","near_ema20"]) and (pb.get("vol_contraction", False))
    flags["pullback_ok"] = 1 if pullback_ok else 0

    weights = {"trend_up":2.0,"pullback_ok":2.0,"momentum_pos":1.0,"no_bear_div":1.0,"rr_tp2_ge_2_5":2.0,"sr_clearance":2.0}
    score = sum(weights[k]*flags.get(k,0) for k in weights)
    tag = "ENTER" if score>=7 else ("WAIT" if score>=5 else "AVOID")
    bucket = "A" if score>=7 else ("B" if score>=5 else "C")
    reasons = [f"{k}={v}" for k,v in flags.items()]
    hints = {"break": nearest_up, "retest": "EMA20" if pb.get("to_ma_tag")!="touch" else None}
    return FilterResult(symbol=struct["symbol"], score=round(score,2), tag=tag, bucket=bucket, reasons=reasons, hints=hints)

def rank_all(structs: List[Dict]) -> List[FilterResult]:
    res = [score_symbol(s) for s in structs]
    return sorted(res, key=lambda r: r.score, reverse=True)
