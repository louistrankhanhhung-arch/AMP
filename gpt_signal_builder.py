# gpt_signal_builder.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import json, re, math
from universe import get_precisions_map, round_levels


# =========================
# Helpers
# =========================

def _trim_float(x: float, ndigits: int = 6) -> float:
    if x is None: return x
    # hạn chế số chữ số thập phân
    return float(f"{float(x):.{ndigits}f}")

def parse_json_block(text: str) -> Tuple[bool, Any, str]:
    """
    Lấy khối JSON đầu tiên từ model output.
    - Ưu tiên code block ```json ... ```
    - Fallback: tìm { ... } ngoài cùng
    """
    try:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.M)
        if m:
            return True, json.loads(m.group(1)), ""
        # fallback: cắt JSON object ngoài cùng
        # (đơn giản: lấy từ ký tự { đầu tiên đến } cuối cùng)
        start = text.find("{")
        end   = text.rfind("}")
        if 0 <= start < end:
            return True, json.loads(text[start:end+1]), ""
        return False, None, "No JSON object found."
    except Exception as e:
        return False, None, f"JSON parse error: {e}"

# =========================
# 1) CLASSIFY
# =========================

CLASSIFY_SCHEMA = {
    "type": "object",
    "required": ["symbol", "side", "action", "confidence", "reasons"],
    "properties": {
        "symbol": {"type": "string"},
        "side":   {"type": "string", "enum": ["long","short","none"]},
        "action": {"type": "string", "enum": ["ENTER","WAIT","AVOID"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasons": {"type": "array", "items":{"type":"string"}, "maxItems": 6},
        # gợi ý thêm để debug
        "trigger_hint": {"type":"object"}
    }
}

def build_messages_classify(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    trigger_1h: Optional[Dict[str, Any]] = None,
    images: Optional[Dict[str, str]] = None,  # {"60": url, "240": url, "D": url}
) -> List[Dict[str, Any]]:
    """
    Trả về messages (system+user) cho API. Model sẽ trả JSON theo CLASSIFY_SCHEMA.
    """
    # gói context (giữ gọn)
    ctx = {
        "struct_4h": struct_4h,
        "struct_1d": struct_1d,
        "trigger_1h": trigger_1h or {}
    }

    system = {
        "role": "system",
        "content": (
            "Bạn là một trader kỹ thuật. Nhiệm vụ: PHÂN LOẠI nhanh để quyết định side và hành động.\n"
            "- Dựa trên ảnh chart (1H/4H/1D) và JSON context (structs 4H/1D + trigger_1h).\n"
            "- Chỉ trả về JSON (không giải thích dài dòng).\n"
            "- Quy tắc:\n"
            "  • Nếu cấu trúc 1D/4H đồng pha + 1H có tín hiệu reclaim/breakout hợp lệ → ưu tiên ENTER.\n"
            "  • Nếu 1D/4H mâu thuẫn hoặc 1H chưa xác nhận → WAIT.\n"
            "  • Nếu ngược xu hướng mạnh hoặc rủi ro cao → AVOID.\n"
            "- side: 'long' | 'short' | 'none' (nếu AVOID).\n"
            "- reasons: 3–6 gạch đầu dòng ngắn, ưu tiên các bằng chứng rõ (RSI, BB, MA20/50, cấu trúc HH/HL/LH/LL, phân kỳ, SR...).\n"
            "- confidence: 0..1.\n"
            "Trả JSON theo schema: "
            + json.dumps(CLASSIFY_SCHEMA, ensure_ascii=False)
        )
    }

    user_parts: List[Dict[str, Any]] = [{"type":"text","text":"Dữ liệu vào (context JSON & ảnh):"}]

    # chèn ảnh
    if images:
        for tf_key, url in images.items():
            user_parts.append({"type":"text","text":f"Ảnh {tf_key}:"})
            user_parts.append({"type":"image_url","image_url":{"url":url}})

    # chèn JSON context
    user_parts.append({"type":"text","text":"Context JSON:"})
    user_parts.append({"type":"text","text":json.dumps(ctx, ensure_ascii=False)})

    user = {"role":"user","content": user_parts}

    return [system, user]

# giữ tương thích tên cũ nếu bạn đang gọi build_prompt_classify(struct)
def build_prompt_classify(struct: Dict[str, Any]) -> str:
    return json.dumps({
        "role": "You are a technical classifier. Return concise JSON.",
        "requirements": CLASSIFY_SCHEMA,
        "struct": struct
    }, ensure_ascii=False)

# =========================
# 2) PLAN
# =========================

PLAN_SCHEMA = {
    "type": "object",
    "required": ["symbol","side","entries","stop","tps","eta"],
    "properties": {
        "symbol": {"type":"string"},
        "side":   {"type":"string","enum":["long","short"]},
        "entries":{"type":"array","items":{"type":"number"}, "minItems":1, "maxItems":2},
        "stop":   {"type":"number"},
        "tps":    {"type":"array","items":{"type":"number"}, "minItems":1, "maxItems":5},
        "eta":    {"type":"object"}  # ví dụ {"tp1":"1-3 ngày","tp2":"3-6 ngày","tp3":"1-2 tuần"}
    }
}

def build_messages_plan(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    side: str,
    classify_reasoning: Optional[Dict[str, Any]] = None,
    images: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Trả về messages để model xuất JSON kế hoạch vào lệnh: entries, SL, TP, ETA (không HTML).
    """
    ctx = {
        "struct_4h": struct_4h,
        "struct_1d": struct_1d,
        "decision": {"side": side, "from_classify": classify_reasoning or {}}
    }

    system = {
        "role":"system",
        "content": (
            "Bạn là nhà giao dịch. Hãy lập kế hoạch vào lệnh ngắn gọn theo JSON duy nhất.\n"
            "- Sử dụng ATR/biên độ gần nhất để đặt SL hợp lý; TP bám theo SR/BB/Fibo.\n"
            "- Quy tắc:\n"
            "  • side=long: stop < entries và tps tăng dần.\n"
            "  • side=short: stop > entries và tps giảm dần.\n"
            "  • Tối đa 2 Entry, tối đa 5 TP. Làm tròn đến 6 chữ số thập phân.\n"
            "  • Gợi ý ETA như 1–3 ngày, 3–6 ngày, 1–2 tuần.\n"
            "- Trả về JSON theo schema: "
            + json.dumps(PLAN_SCHEMA, ensure_ascii=False)
        )
    }

    user_parts: List[Dict[str, Any]] = [{"type":"text","text":"Dữ liệu vào (context JSON & ảnh):"}]

    if images:
        for tf_key, url in images.items():
            user_parts.append({"type":"text","text":f"Ảnh {tf_key}:"})
            user_parts.append({"type":"image_url","image_url":{"url":url}})

    user_parts.append({"type":"text","text":"Context JSON:"})
    user_parts.append({"type":"text","text":json.dumps(ctx, ensure_ascii=False)})

    user = {"role":"user","content": user_parts}
    return [system, user]

# tương thích tên cũ (nhưng đã bỏ HTML)
def build_prompt_plan(struct: Dict[str, Any]) -> str:
    return json.dumps({
        "role": "You are a strategist. Produce Entries/SL/TP/ETA using ATR.",
        "requirements": PLAN_SCHEMA,
        "struct": struct
    }, ensure_ascii=False)

# =========================
# 3) VALIDATOR & PARSER
# =========================

def validate_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    side = plan.get("side")
    entries = plan.get("entries") or []
    tps = plan.get("tps") or []
    stop = plan.get("stop")

    if side not in ("long","short"):
        errs.append("side must be long|short")

    if not isinstance(entries, list) or not entries:
        errs.append("entries must be 1-2 numbers")
    if not isinstance(tps, list) or not tps:
        errs.append("tps must be 1-5 numbers")
    if not isinstance(stop, (int,float)):
        errs.append("stop must be number")

    try:
        entries = [_trim_float(float(x)) for x in entries]
        tps     = [_trim_float(float(x)) for x in tps]
        stop    = _trim_float(float(stop))
    except Exception:
        errs.append("entries/stop/tps must be numeric")

    if not errs and side in ("long","short"):
        if side == "long":
            if not (stop < min(entries)):
                errs.append("For long: stop must be < min(entries)")
            if any(tps[i] >= tps[i+1] for i in range(len(tps)-1)):
                errs.append("For long: tps must be strictly increasing")
        else:
            if not (stop > max(entries)):
                errs.append("For short: stop must be > max(entries)")
            if any(tps[i] <= tps[i+1] for i in range(len(tps)-1)):
                errs.append("For short: tps must be strictly decreasing")

        # không cho giá âm/0
        if any(x <= 0 for x in entries + [stop] + tps):
            errs.append("All price levels must be > 0")

        # giới hạn số thập phân
        if any(len(str(x).split(".")[-1]) > 8 for x in entries + [stop] + tps if isinstance(x, float)):
            errs.append("Too many decimals (>8)")

    return (len(errs) == 0), errs

def parse_plan_output(text: str) -> Dict[str, Any]:
    ok, obj, err = parse_json_block(text)
    if not ok:
        return {"ok": False, "error": err}

    valid, errs = validate_plan(obj)
    if not valid:
        return {"ok": False, "error": "; ".join(errs), "raw": obj}

    # chuẩn hoá số
    obj["entries"] = [_trim_float(x) for x in obj["entries"]]
    obj["stop"]    = _trim_float(obj["stop"])
    obj["tps"]     = [_trim_float(x) for x in obj["tps"]]
    return {"ok": True, "plan": obj}

def apply_exchange_rounding(plan: dict, exchange: str = "KUCOIN") -> dict:
    try:
        mp = get_precisions_map(exchange, [plan["symbol"]])
        dp = mp[plan["symbol"]]["price_dp"]
        return round_levels(plan, dp)
    except Exception:
        return plan

parsed = parse_plan_output(model_text)
if parsed["ok"]:
    plan = parsed["plan"]
    plan = apply_exchange_rounding(plan, exchange="KUCOIN")
    # -> lưu/hiển thị plan đã làm tròn

# =========================
# (Tuỳ chọn) tiện ích đóng gói
# =========================

def make_images_payload(s3_or_public_urls: Dict[str,str]) -> Dict[str,str]:
    """
    Chuẩn hoá map TF -> URL ảnh để gắn vào messages.
    Keys chấp nhận: '60','240','D' hoặc '1H','4H','1D'.
    """
    out = {}
    for k,v in (s3_or_public_urls or {}).items():
        kk = k.upper()
        if kk in ("60","1H"): out["60"] = v
        elif kk in ("240","4H"): out["240"] = v
        elif kk in ("D","1D"): out["D"] = v
    return out
