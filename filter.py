
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class FilterResult:
    symbol: str
    score: float
    tag: str
    bucket: str
    reasons: list
    hints: dict
    # --- Added short-side fields (backward compatible: optional) ---
    score_short: Optional[float] = None
    tag_short: Optional[str] = None
    bucket_short: Optional[str] = None
    reasons_short: Optional[list] = None
    hints_short: Optional[dict] = None
    # --- Best-of-direction summary ---
    direction_best: Optional[str] = None   # 'LONG' | 'SHORT'
    score_best: Optional[float] = None
    tag_best: Optional[str] = None
    bucket_best: Optional[str] = None

def _nearest_above(levels: List[float], px: float) -> Optional[float]:
    ups = sorted([lv for lv in levels if lv > px])
    return ups[0] if ups else None

def _nearest_below(levels: List[float], px: float) -> Optional[float]:
    dns = sorted([lv for lv in levels if lv < px], reverse=True)
    return dns[0] if dns else None

def _bucket_from_score(score: float) -> str:
    return "A" if score >= 7 else ("B" if score >= 5 else "C")

def _tag_from_score(score: float) -> str:
    return "ENTER" if score >= 7 else ("WAIT" if score >= 5 else "AVOID")

def _calc_rr(close: float, sl: float, tp: float) -> float:
    risk = abs(close - sl)
    return (abs(tp - close) / risk) if risk > 0 else 0.0

def score_symbol(struct: Dict) -> FilterResult:
    snap = struct.get("snapshot", {}) or {}
    lvl  = struct.get("levels", {}) or {}
    trg  = struct.get("targets", {}) or {}
    st   = struct.get("structure", {}) or {}
    div  = struct.get("divergence", {}) or {}
    ctxg = struct.get("context_guidance", {}) or {}

    close = float(((snap.get("price") or {}).get("close")) or 0.0)
    atr   = float(snap.get("atr14") or 0.0)
    rsi   = float(snap.get("rsi14") or 50.0)
    trend = (st.get("trend") or {}).get("state")

    sr_up = list(lvl.get("sr_up") or [])
    sr_dn = list(lvl.get("sr_down") or [])

    # ---------- LONG scoring (existing behavior) ----------
    flags_long = {}
    nearest_up = _nearest_above(sr_up, close) if sr_up else None
    dist_up_atr = ((nearest_up - close) / atr) if (nearest_up and atr > 0) else 0.0
    flags_long["trend_up"] = 1 if trend == "up" else 0
    flags_long["momentum_pos"] = 1 if rsi > 50 else 0
    flags_long["no_bear_div"] = 1 if (div.get("rsi_price") != "bearish") else 0
    flags_long["sr_clearance"] = 1 if dist_up_atr > 0.7 else 0

    # TP2 (ưu tiên targets.up_bands[1], fallback sr_up[1])
    tp2 = None
    if (trg.get("up_bands") or []) and len(trg["up_bands"]) >= 2:
        tp2 = float(trg["up_bands"][1]["tp"])
    elif len(sr_up) >= 2:
        tp2 = float(sr_up[1])

    sl_long = close - 1.2 * atr if atr > 0 else close * 0.98
    rr_tp2 = _calc_rr(close, sl_long, tp2) if tp2 else 0.0
    flags_long["rr_tp2_ge_2_5"] = 1 if rr_tp2 >= 2.5 else 0

    pb = st.get("pullback") or {}
    pullback_ok_long = (pb.get("to_ma_tag") in ["touch","near_ema20"]) and bool(pb.get("vol_healthy") or pb.get("vol_contraction"))
    flags_long["pullback_ok"] = 1 if pullback_ok_long else 0

    weights = {"trend_up":2.0,"pullback_ok":2.0,"momentum_pos":1.0,"no_bear_div":1.0,"rr_tp2_ge_2_5":2.0,"sr_clearance":2.0}
    score_long = sum(weights[k]*flags_long.get(k,0) for k in weights)
    tag_long = _tag_from_score(score_long)
    bucket_long = _bucket_from_score(score_long)
    reasons_long = [f"{k}={v}" for k,v in flags_long.items()]
    hints_long = {"break": nearest_up, "retest": "EMA20" if pb.get("to_ma_tag")!="touch" else None}

    # ---------- SHORT scoring (new) ----------
    flags_short = {}
    nearest_dn = _nearest_below(sr_dn, close) if sr_dn else None
    dist_dn_atr = ((close - nearest_dn) / atr) if (nearest_dn and atr > 0) else 0.0
    flags_short["trend_down"] = 1 if trend == "down" else 0
    flags_short["momentum_neg"] = 1 if rsi < 50 else 0
    flags_short["no_bull_div"] = 1 if (div.get("rsi_price") != "bullish") else 1  # current engine doesn't tag 'bullish'

    flags_short["sr_clearance_down"] = 1 if dist_dn_atr > 0.7 else 0

    # TP2 short (ưu tiên targets.down_bands[1], fallback sr_down[1])
    tp2d = None
    if (trg.get("down_bands") or []) and len(trg["down_bands"]) >= 2:
        tp2d = float(trg["down_bands"][1]["tp"])
    elif len(sr_dn) >= 2:
        tp2d = float(sr_dn[-2])  # second closest down could be ambiguous; use second element from bottom

    sl_short = close + 1.2 * atr if atr > 0 else close * 1.02
    rr_tp2d = _calc_rr(close, sl_short, tp2d) if tp2d else 0.0
    flags_short["rr_tp2_ge_2_5_down"] = 1 if rr_tp2d >= 2.5 else 0

    pullback_ok_short = (pb.get("to_ma_tag") in ["touch","near_ema20"]) and bool(pb.get("vol_healthy") or pb.get("vol_contraction"))
    flags_short["pullback_ok_down"] = 1 if pullback_ok_short else 0

    weights_s = {"trend_down":2.0,"pullback_ok_down":2.0,"momentum_neg":1.0,"no_bull_div":1.0,"rr_tp2_ge_2_5_down":2.0,"sr_clearance_down":2.0}
    score_short = sum(weights_s[k]*flags_short.get(k,0) for k in weights_s)
    tag_short = _tag_from_score(score_short)
    bucket_short = _bucket_from_score(score_short)
    reasons_short = [f"{k}={v}" for k,v in flags_short.items()]
    hints_short = {"break": nearest_dn, "retest": "EMA20"}  # for short, prefer retest toward EMA20 as well

    # ---------- Best-of-direction summary ----------
    if score_short > score_long:
        direction_best, score_best, tag_best, bucket_best = "SHORT", score_short, tag_short, bucket_short
    else:
        direction_best, score_best, tag_best, bucket_best = "LONG", score_long, tag_long, bucket_long

    # Return with original (long) fields preserved
    return FilterResult(
        symbol=struct.get("symbol","?"),
        score=round(score_long,2),
        tag=tag_long,
        bucket=bucket_long,
        reasons=reasons_long,
        hints=hints_long,
        score_short=round(score_short,2),
        tag_short=tag_short,
        bucket_short=bucket_short,
        reasons_short=reasons_short,
        hints_short=hints_short,
        direction_best=direction_best,
        score_best=round(score_best,2),
        tag_best=tag_best,
        bucket_best=bucket_best,
    )

def rank_all(structs: List[Dict]) -> List[FilterResult]:
    res = [score_symbol(s) for s in structs]
    # sort by best score if available, otherwise by long score
    return sorted(res, key=lambda r: (r.score_best if r.score_best is not None else r.score), reverse=True)
