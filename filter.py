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

# ---- helpers (mới) ---------------------------------------------------------

def _get(d, *keys, default=None):
    cur = d or {}
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k)
    return default if cur is None else cur

def _safe_float(x, fallback=0.0):
    try:
        return float(x)
    except Exception:
        return float(fallback)

def _is_extended_long(close: float, ema20: float, atr: float) -> bool:
    # Nới tiêu chí extended để breakout không bị loại oan
    if atr <= 0: return False
    return (close - ema20) / atr > 1.2

def _is_extended_short(close: float, ema20: float, atr: float) -> bool:
    if atr <= 0: return False
    return (ema20 - close) / atr > 1.2

def _rsi_soft_score_long(rsi4: float) -> int:
    """
    RSI nóng -> penalty mềm, KHÔNG loại.
    55–70: +2  |  70–80: -1  |  >80: -2 (vẫn có thể ENTER nếu breakout_ok)
    """
    if 55 <= rsi4 <= 70: return 2
    if 70 < rsi4 <= 80: return -1
    if rsi4 > 80: return -2
    if 48 <= rsi4 < 55: return 1  # mép reclaim
    return 0

def _rsi_soft_score_short(rsi4: float) -> int:
    """
    Đối xứng cho short.
    30–45: +2  |  20–30: -1  |  <20: -2
    """
    if 30 <= rsi4 <= 45: return 2
    if 20 < rsi4 < 30: return -1
    if rsi4 <= 20: return -2
    if 45 < rsi4 <= 52: return 1  # mép reclaim_down
    return 0

# ---------------------------------------------------------------------------

def score_symbol(struct: Dict) -> FilterResult:
    snap = struct.get("snapshot", {}) or {}
    lvl  = struct.get("levels", {}) or {}
    st   = struct.get("structure", {}) or {}
    ev   = struct.get("events", {}) or {}
    conf = struct.get("confirmations", {}) or {}

    trend_state = _get(st, "trend", "state", default="side")  # "up" | "down" | "sideways"
    rsi = _safe_float(snap.get("rsi14"), 50.0)
    atr = _safe_float(snap.get("atr14"), 1.0) or 1.0
    price = snap.get("price", {}) or {}
    close = _safe_float(price.get("close"), 0.0)

    ema20 = _safe_float(_get(snap, "ma", "ema20"), 0.0)
    ema50 = _safe_float(_get(snap, "ma", "ema50"), 0.0)

    # Levels
    sr_up   = lvl.get("sr_up", []) or []     # kháng cự phía trên (tăng dần)
    sr_down = lvl.get("sr_down", []) or []   # hỗ trợ phía dưới (tăng dần)

    # Divergence
    div = _get(struct, "divergence", "rsi_price", default="none")  # "bearish" | "bullish" | "none"

    # Pullback info (đồng bộ tên khóa với structure_engine: vol_healthy)
    pb = st.get("pullback", {}) or {}
    # vol_contraction cũ vẫn giữ an toàn; ưu tiên vol_healthy nếu có
    pullback_vol_healthy = bool(pb.get("vol_healthy")) or bool(pb.get("vol_contraction"))
    pullback_ok = (pb.get("to_ma_tag") in ["touch", "near_ema20"]) and pullback_vol_healthy

    # Side hint đối xứng
    side_hint = "long" if (trend_state == "up" and rsi >= 50) else ("short" if (trend_state == "down" and rsi <= 50) else None)

    # Breakout/breakdown + volume xác nhận (ưu tiên cao)
    vol_conf = conf.get("volume", {}) or {}
    breakout_ok  = bool(ev.get("last_breakout_confirmed") or ev.get("breakout")) and (bool(vol_conf.get("breakout_vol_ok")) or bool(vol_conf.get("breakout_vol_healthy")) or _safe_float(vol_conf.get("vol_z20"), 0) >= 1.0)
    breakdown_ok = bool(ev.get("last_breakdown_confirmed") or ev.get("breakdown")) and (bool(vol_conf.get("breakdown_vol_ok")) or bool(vol_conf.get("breakdown_vol_healthy")) or _safe_float(vol_conf.get("vol_z20"), 0) >= 1.0)

    flags, weights = {}, {}
    reasons_extra = []

    if side_hint == "long":
        # Clearance lên (ATR)
        up_above = [x for x in sr_up if _safe_float(x) > close]
        nearest_up = min(up_above) if up_above else None
        dist_up_atr = (nearest_up - close) / atr if (nearest_up and atr) else 0.0

        # R:R tới TP2
        rr_tp2 = 0.0
        if len(up_above) >= 2:
            tp2 = sorted(up_above)[1]
            sl  = close - 1.2 * atr
            denom = (close - sl)
            rr_tp2 = (tp2 - close) / denom if denom != 0 else 0.0

        # Soft scores
        rsi_soft = _rsi_soft_score_long(rsi)
        extended = _is_extended_long(close, ema20, atr)

        flags["trend_up"]        = 1
        flags["momentum_ok"]     = 1 if rsi >= 48 else 0     # nới mép reclaim
        flags["pullback_ok"]     = 1 if pullback_ok else 0
        flags["no_bear_div"]     = 1 if div != "bearish" else 0
        flags["sr_clearance_up"] = 1 if dist_up_atr > 0.7 else 0
        flags["rr_tp2_ge_2_5"]   = 1 if rr_tp2 >= 2.5 else 0
        flags["breakout_ok"]     = 1 if breakout_ok else 0
        flags["extended"]        = 1 if extended else 0
        flags["rsi_soft"]        = rsi_soft  # có thể âm/dương

        weights = {
            "trend_up":2.0, "pullback_ok":2.0, "momentum_ok":1.0,
            "no_bear_div":1.0, "rr_tp2_ge_2_5":2.0, "sr_clearance_up":2.0,
            "breakout_ok":3.5,  # ↑ ưu tiên breakout thực sự
            # soft terms sẽ cộng trực tiếp theo giá trị (+2, +1, -1, -2)
        }

        reasons_extra = [
            f"dist_up_atr={round(dist_up_atr,2)}",
            f"rr_tp2={round(rr_tp2,2)}",
            f"rsi_soft={rsi_soft}",
            f"extended={extended}",
            f"breakout_ok={breakout_ok}"
        ]

    elif side_hint == "short":
        dn_below = [x for x in sr_down if _safe_float(x) < close]
        nearest_dn = max(dn_below) if dn_below else None
        dist_dn_atr = (close - nearest_dn) / atr if (nearest_dn and atr) else 0.0

        rr_tp2 = 0.0
        if len(dn_below) >= 2:
            tp2 = sorted(dn_below)[-2]
            sl  = close + 1.2 * atr
            denom = (sl - close)
            rr_tp2 = (close - tp2) / denom if denom != 0 else 0.0

        rsi_soft = _rsi_soft_score_short(rsi)
        extended = _is_extended_short(close, ema20, atr)

        flags["trend_down"]        = 1
        flags["momentum_ok"]       = 1 if rsi <= 52 else 0   # nới mép reclaim_down
        flags["pullback_ok"]       = 1 if pullback_ok else 0
        flags["no_bull_div"]       = 1 if div != "bullish" else 0
        flags["sr_clearance_down"] = 1 if dist_dn_atr > 0.7 else 0
        flags["rr_tp2_ge_2_5"]     = 1 if rr_tp2 >= 2.5 else 0
        flags["breakdown_ok"]      = 1 if breakdown_ok else 0
        flags["extended"]          = 1 if extended else 0
        flags["rsi_soft"]          = rsi_soft

        weights = {
            "trend_down":2.0, "pullback_ok":2.0, "momentum_ok":1.0,
            "no_bull_div":1.0, "rr_tp2_ge_2_5":2.0, "sr_clearance_down":2.0,
            "breakdown_ok":3.5,
        }

        reasons_extra = [
            f"dist_dn_atr={round(dist_dn_atr,2)}",
            f"rr_tp2={round(rr_tp2,2)}",
            f"rsi_soft={rsi_soft}",
            f"extended={extended}",
            f"breakdown_ok={breakdown_ok}"
        ]

    else:
        # Trung tính: không ưu tiên bên nào -> điểm thấp, rơi WAIT/AVOID như cũ
        flags = {"neutral": 1 if trend_state == "sideways" else 0}
        weights = {"neutral": 0.5}
        reasons_extra = ["side_hint=None"]

    # Tính điểm: phần “soft” (rsi_soft) cộng trực tiếp
    score = sum(weights.get(k,0)* (flags.get(k,0) if k != "rsi_soft" else 0) for k in weights)
    score += flags.get("rsi_soft", 0)

    # Penalty mềm cho extended (không loại hẳn)
    if flags.get("extended"):
        score -= 1.0

    # Tag theo ngưỡng, nhưng NHẤN MẠNH breakout:
    tag = "ENTER" if score >= 7 else ("WAIT" if score >= 5 else "AVOID")
    bucket = "A" if score >= 7 else ("B" if score >= 5 else "C")

    # Nếu breakout/breakdown thật sự có volume → tối thiểu phải là WAIT
    if side_hint == "long" and breakout_ok and tag == "AVOID":
        tag, bucket = "WAIT", "B"
    if side_hint == "short" and breakdown_ok and tag == "AVOID":
        tag, bucket = "WAIT", "B"

    # Nếu trend lớn đồng pha + breakout_ok → cho phép “đẩy” từ WAIT lên ENTER khi score sát ngưỡng
    if side_hint == "long" and breakout_ok and tag == "WAIT" and score >= 6.5:
        tag, bucket = "ENTER", "A"
    if side_hint == "short" and breakdown_ok and tag == "WAIT" and score >= 6.5:
        tag, bucket = "ENTER", "A"

    reasons = [f"{k}={v}" for k, v in flags.items()] + reasons_extra
    hints = {
        "side_hint": side_hint,
        "retest": "EMA20" if pb.get("to_ma_tag") != "touch" else None
    }
    return FilterResult(
        symbol=struct.get("symbol","?"),
        score=round(score, 2),
        tag=tag,
        bucket=bucket,
        reasons=reasons,
        hints=hints
    )

def rank_all(structs: List[Dict]) -> List[FilterResult]:
    res = [score_symbol(s) for s in structs]
    return sorted(res, key=lambda r: r.score, reverse=True)