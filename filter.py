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
    st  = struct.get("structure", {})
    trend = st.get("trend", {}).get("state","side")      # "up" | "down" | "sideways"
    rsi = float(snap.get("rsi14", 50))
    atr = float(snap.get("atr14", 1) or 1)
    close = float(snap.get("price",{}).get("close", 0) or 0)

    # Levels
    sr_up   = lvl.get("sr_up", [])    # kháng cự phía trên (list giá tăng dần)
    sr_down = lvl.get("sr_down", [])  # hỗ trợ phía dưới (list giá tăng dần) - có thể không có, xử lý an toàn

    # Divergence
    div = struct.get("divergence", {}).get("rsi_price", "none")  # "bearish" | "bullish" | "none"

    # Pullback info
    pb = st.get("pullback", {})
    pullback_ok = (pb.get("to_ma_tag") in ["touch","near_ema20"]) and bool(pb.get("vol_contraction", False))

    # Xác định hướng thiên (hint) để chấm điểm đối xứng
    side_hint = "long" if (trend == "up" and rsi >= 50) else ("short" if (trend == "down" and rsi <= 50) else None)

    flags, weights = {}, {}

    if side_hint == "long":
        # Clearance lên: khoảng cách tới kháng cự gần nhất phía trên (ATR)
        up_above = [x for x in sr_up if x > close]
        nearest_up = min(up_above) if up_above else None
        dist_up_atr = (nearest_up - close) / atr if (nearest_up and atr) else 0

        # R:R tới TP2: lấy kháng cự thứ 2 phía trên nếu có
        rr_tp2 = 0.0
        if len(up_above) >= 2:
            tp2 = sorted(up_above)[1]
            sl  = close - 1.2 * atr
            denom = (close - sl)
            rr_tp2 = (tp2 - close) / denom if denom != 0 else 0.0

        flags["trend_up"]        = 1
        flags["momentum_ok"]     = 1 if rsi >= 50 else 0
        flags["pullback_ok"]     = 1 if pullback_ok else 0
        flags["no_bear_div"]     = 1 if div != "bearish" else 0
        flags["sr_clearance_up"] = 1 if dist_up_atr > 0.7 else 0
        flags["rr_tp2_ge_2_5"]   = 1 if rr_tp2 >= 2.5 else 0

        weights = {
            "trend_up":2.0, "pullback_ok":2.0, "momentum_ok":1.0,
            "no_bear_div":1.0, "rr_tp2_ge_2_5":2.0, "sr_clearance_up":2.0
        }

        reasons_extra = [f"dist_up_atr={round(dist_up_atr,2)}", f"rr_tp2={round(rr_tp2,2)}"]

    elif side_hint == "short":
        # Clearance xuống: khoảng cách tới hỗ trợ gần nhất phía dưới (ATR)
        dn_below = [x for x in sr_down if x < close]
        nearest_dn = max(dn_below) if dn_below else None
        dist_dn_atr = (close - nearest_dn) / atr if (nearest_dn and atr) else 0

        # R:R tới TP2: lấy hỗ trợ thứ 2 phía dưới nếu có
        rr_tp2 = 0.0
        if len(dn_below) >= 2:
            # hỗ trợ phía dưới sắp xếp tăng dần -> phần tử -2 là hỗ trợ thứ 2 dưới close
            tp2 = sorted(dn_below)[-2]
            sl  = close + 1.2 * atr
            denom = (sl - close)
            rr_tp2 = (close - tp2) / denom if denom != 0 else 0.0

        flags["trend_down"]         = 1
        flags["momentum_ok"]        = 1 if rsi <= 50 else 0
        flags["pullback_ok"]        = 1 if pullback_ok else 0
        flags["no_bull_div"]        = 1 if div != "bullish" else 0
        flags["sr_clearance_down"]  = 1 if dist_dn_atr > 0.7 else 0
        flags["rr_tp2_ge_2_5"]      = 1 if rr_tp2 >= 2.5 else 0

        weights = {
            "trend_down":2.0, "pullback_ok":2.0, "momentum_ok":1.0,
            "no_bull_div":1.0, "rr_tp2_ge_2_5":2.0, "sr_clearance_down":2.0
        }

        reasons_extra = [f"dist_dn_atr={round(dist_dn_atr,2)}", f"rr_tp2={round(rr_tp2,2)}"]

    else:
        # Trung tính: không ưu tiên bên nào -> điểm thấp, rơi WAIT/AVOID như cũ
        flags = {"neutral": 1 if trend == "sideways" else 0}
        weights = {"neutral": 0.5}
        reasons_extra = ["side_hint=None"]

    score = sum(weights[k]*flags.get(k,0) for k in weights)

    tag = "ENTER" if score >= 7 else ("WAIT" if score >= 5 else "AVOID")
    bucket = "A" if score >= 7 else ("B" if score >= 5 else "C")

    reasons = [f"{k}={v}" for k, v in flags.items()] + reasons_extra
    hints = {
        "side_hint": side_hint,
        "retest": "EMA20" if pb.get("to_ma_tag") != "touch" else None
    }
    return FilterResult(
        symbol=struct["symbol"],
        score=round(score, 2),
        tag=tag,
        bucket=bucket,
        reasons=reasons,
        hints=hints
    )

def rank_all(structs: List[Dict]) -> List[FilterResult]:
    res = [score_symbol(s) for s in structs]
    return sorted(res, key=lambda r: r.score, reverse=True)
