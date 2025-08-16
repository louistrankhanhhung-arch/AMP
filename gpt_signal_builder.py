from __future__ import annotations

"""Utilities to classify and plan trades, and build Telegram-ready signals.

This module orchestrates two chat completions (classify → plan), validates the
returned JSON, computes R:R, and formats output for Telegram plus an internal
Vietnamese analysis block for logs.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import os
import re

from openai import OpenAI

# ---------------------------------------------------------------------------
# Defaults & constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEFAULT_RISK_MODE = os.getenv("RISK_MODE", "conservative").lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_json_block(text: str) -> Tuple[bool, Any, str]:
    """Extract and parse a JSON object from a model response.

    Returns:
        (ok, obj, err)
    """
    try:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.M)
        if m:
            return True, json.loads(m.group(1)), ""

        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            return True, json.loads(text[start : end + 1]), ""
        return False, None, "No JSON object found."
    except Exception as e:  # noqa: BLE001 - keep broad to surface any JSON issue
        return False, None, f"JSON parse error: {e}"


def _trim_float(x: Optional[float], ndigits: int = 6) -> Optional[float]:
    if x is None:
        return x
    return float(f"{float(x):.{ndigits}f}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
CLASSIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["symbol", "side", "action", "confidence", "reasons"],
    "properties": {
        "symbol": {"type": "string"},
        "side": {"type": "string", "enum": ["long", "short", "none"]},
        "action": {"type": "string", "enum": ["ENTER", "WAIT", "AVOID"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "trigger_hint": {"type": "object"},
    },
}

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["symbol", "side", "entries", "stop", "tps", "eta"],
    "properties": {
        "symbol": {"type": "string"},
        "side": {"type": "string", "enum": ["long", "short"]},
        "entries": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1,
            "maxItems": 2,
        },
        "stop": {"type": "number"},
        "tps": {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 3},
        "eta": {"type": "object"},
    },
}


# ---------------------------------------------------------------------------
# Prompt builders (classify / plan)
# ---------------------------------------------------------------------------

def build_messages_classify(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    trigger_1h: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    ctx = {"struct_4h": struct_4h, "struct_1d": struct_1d, "trigger_1h": trigger_1h or {}}
    system = {
        "role": "system",
        "content": (
            "Bạn là trader kỹ thuật. Nhiệm vụ: PHÂN LOẠI nhanh side và hành động dựa trên JSON context (4H/1D + trigger_1h). "
            "Chỉ trả về JSON **tiếng Việt** theo schema: "
            + json.dumps(CLASSIFY_SCHEMA, ensure_ascii=False)
            + "\nQuy tắc: 1D/4H đồng pha + 1H xác nhận → ENTER; mâu thuẫn/chưa rõ → WAIT; ngược mạnh → AVOID. "
            + "Nếu 4H/1D cùng giảm (hoặc cùng tăng) nhưng 1H chưa xác nhận, đặt action='WAIT' (không dùng AVOID). "
              "AVOID chỉ khi định đi ngược xu hướng chính hoặc R:R xấu/levels tắc. "
            "Trường 'reasons' là mảng 3–6 câu **tiếng Việt**, ngắn gọn, liên hệ trực tiếp các số liệu (RSI, EMA20/50, BB, swing...)."
        ),
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Context JSON:"},
            {"type": "text", "text": json.dumps(ctx, ensure_ascii=False)},
        ],
    }
    return [system, user]


def _sl_policy_block(risk_mode: str) -> str:
    rm = (risk_mode or DEFAULT_RISK_MODE).lower()
    return (
        f"""Chính sách SL = {rm}.
- Nếu conservative:
  • LONG: đặt SL **bảo thủ** để tránh wick: SL = min(đáy swing gần (4H, 5–7 nến), EMA50_4H) − 0.2×ATR14_4H (nếu có).
  • SHORT: SL = max(đỉnh swing gần (4H, 5–7 nến), EMA50_4H) + 0.2×ATR14_4H.
  • Thiếu dữ liệu → ưu tiên EMA50_4H và swing gần nhất; KHÔNG dùng EMA20 làm SL.
  • SL xét theo **đóng nến 4H** để hạn chế wick.
- Nếu neutral: dùng swing gần nhất ± 0.1×ATR14_4H.
- Nếu aggressive: tham chiếu EMA20_4H ± 0.1×ATR14_4H.
BẮT BUỘC tuân thủ chính sách SL ở trên; chỉ xuất **1 giá trị** 'stop' trong JSON."""
    ).strip()


def build_messages_plan(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    side: str,
    classify_reasoning: Optional[Dict[str, Any]] = None,
    risk_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ctx = {
        "struct_4h": struct_4h,
        "struct_1d": struct_1d,
        "decision": {"side": side, "from_classify": classify_reasoning or {}},
    }
    system = {
        "role": "system",
        "content": (
            "Bạn là nhà giao dịch. Hãy lập kế hoạch vào lệnh **bằng tiếng Việt**, trả duy nhất JSON theo schema: "
            + json.dumps(PLAN_SCHEMA, ensure_ascii=False)
            + "\nQuy tắc chung: long → stop < entries, TP tăng dần; short → stop > entries, TP giảm dần. "
            "Tối đa 2 Entry, tối đa 3 TP. Ưu tiên mức SR/BB/đỉnh-đáy gần, tính hợp lý và an toàn. "
            + _sl_policy_block(risk_mode or DEFAULT_RISK_MODE)
            + "\nKhông xuất HTML, không thêm văn bản ngoài JSON."
        ),
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Context JSON:"},
            {"type": "text", "text": json.dumps(ctx, ensure_ascii=False)},
        ],
    }
    return [system, user]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    side = plan.get("side")
    entries = plan.get("entries") or []
    tps = plan.get("tps") or []
    stop = plan.get("stop")

    if side not in ("long", "short"):
        errs.append("side must be long|short")
    if not isinstance(entries, list) or not entries:
        errs.append("entries must be 1-2 numbers")
    if not isinstance(tps, list) or not tps:
        errs.append("tps must be 1-3 numbers")
    if not isinstance(stop, (int, float)):
        errs.append("stop must be number")

    try:
        entries = [_trim_float(float(x)) for x in entries]
        tps = [_trim_float(float(x)) for x in tps]
        stop = _trim_float(float(stop))  # type: ignore[arg-type]
    except Exception:
        errs.append("entries/stop/tps must be numeric")

    if not errs and side in ("long", "short"):
        if side == "long":
            if not (stop < min(entries)):
                errs.append("For long: stop < min(entries)")
            if any(tps[i] >= tps[i + 1] for i in range(len(tps) - 1)):
                errs.append("For long: tps strictly increasing")
        else:
            if not (stop > max(entries)):
                errs.append("For short: stop > max(entries)")
            if any(tps[i] <= tps[i + 1] for i in range(len(tps) - 1)):
                errs.append("For short: tps strictly decreasing")
        if any(x <= 0 for x in entries + [stop] + tps):  # type: ignore[list-item]
            errs.append("All price levels must be > 0")

    return (len(errs) == 0), errs


def parse_plan_output(text: str) -> Dict[str, Any]:
    ok, obj, err = parse_json_block(text)
    if not ok:
        return {"ok": False, "error": err}

    valid, errs = validate_plan(obj)
    if not valid:
        return {"ok": False, "error": "; ".join(errs), "raw": obj}

    obj["entries"] = [_trim_float(x) for x in obj["entries"]]
    obj["stop"] = _trim_float(obj["stop"])  # type: ignore[index]
    obj["tps"] = [_trim_float(x) for x in obj["tps"]][:3]
    return {"ok": True, "plan": obj}


# ---------------------------------------------------------------------------
# RR & leverage
# ---------------------------------------------------------------------------

def pick_leverage(conf: float) -> str:
    if conf >= 0.80:
        return "x10"
    if conf >= 0.65:
        return "x5"
    return "x3"


def compute_rr(side: str, entries: List[float], stop: float, tps: List[float]) -> Dict[str, Any]:
    avg_entry = sum(entries) / len(entries)
    rr_list: List[Optional[float]] = []
    for tp in tps:
        if side == "long":
            risk = avg_entry - stop
            reward = tp - avg_entry
        else:
            risk = stop - avg_entry
            reward = avg_entry - tp
        rr_list.append(None if risk <= 0 else reward / risk)

    non_none = [x for x in rr_list if x is not None]
    return {
        "avg_entry": avg_entry,
        "rr_list": rr_list,
        "rr_min": min(non_none, default=None),
        "rr_max": max(non_none, default=None),
    }


# ---------------------------------------------------------------------------
# Rounding by exchange (optional)
# ---------------------------------------------------------------------------

def round_by_exchange(plan: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from universe import get_precisions_map, round_levels  # type: ignore

        mp = get_precisions_map("KUCOIN", [plan["symbol"]])
        dp = mp[plan["symbol"]]["price_dp"]
        return round_levels(plan, dp)
    except Exception:
        return plan


# ---------------------------------------------------------------------------
# Analysis prompt (Vietnamese, detailed for logs)
# ---------------------------------------------------------------------------

def build_messages_analysis(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    trigger_1h: Optional[Dict[str, Any]],
    decision: Dict[str, Any],
    plan: Optional[Dict[str, Any]],
    rr_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Yêu cầu GPT viết phân tích chi tiết (tiếng Việt), dùng SỐ LIỆU trong JSON:
    - Xu hướng 1D/4H (trạng thái so với EMA20/EMA50, độ dốc BB)
    - RSI (1D/4H, nếu có)
    - Volume vs SMA20 (nếu có)
    - SR chính (Daily) và SR nội ngày (4H) từ context_levels.* nếu có
    - Những vùng mỏng/thanh khoản yếu (nếu có trường liquidity)
    - Trigger 1H: reclaim/pullback (nếu có)
    - Kế hoạch hành động ngắn gọn (enter, quản trị sau TP1, điều kiện thoát sớm)
    - R:R (rr_min/rr_max) & ETA (tóm tắt)
    """

    sys = {
        "role": "system",
        "content": (
            "Bạn là trợ lý giao dịch. Hãy viết phân tích **TIẾNG VIỆT**, súc tích, dựa 100% vào số liệu trong JSON. "
            "Không nói chung chung. Nếu thiếu dữ liệu, bỏ qua mục đó."
        ),
    }

    payload = {
        "struct_4h": struct_4h,
        "struct_1d": struct_1d,
        "trigger_1h": trigger_1h or {},
        "decision": decision or {},
        "plan": plan or {},
        "rr": rr_meta or {},
    }

    usr = {
        "role": "user",
        "content": (
            "Viết phần **PHÂN TÍCH CHI TIẾT** để in Logs (không gửi Telegram). Dàn ý:\n"
            "• Xu hướng: 1D/4H đang ở đâu so với EMA20/EMA50, BB dốc thế nào.\n"
            "• RSI: giá trị hiện tại 1D/4H (nếu có), còn dư địa hay quá mua.\n"
            "• Volume: so với SMA20 (nếu có) – tăng/giảm.\n"
            "• Hỗ trợ/kháng cự: liệt kê 2–4 mức gần nhất từ context_levels (Daily & 4H) nếu có.\n"
            "• Vùng mỏng/thanh khoản: nếu có liquidity_zones.\n"
            "• Trigger 1H (nếu có): reclaim MA20/pullback…\n"
            "• Kế hoạch: tóm tắt cách vào/thoát, điều kiện dời SL/thoát sớm.\n"
            "• R:R & ETA: tóm tắt rr_min/rr_max và ETA ngắn gọn.\n"
            "YÊU CẦU: Nêu số liệu cụ thể (ví dụ: RSI=56, Close>EMA20≈24.5). Không thêm emoji, không nhắc chuyện gửi Telegram."
            f"\nJSON:\n{json.dumps(payload, ensure_ascii=False)}"
        ),
    }
    return [sys, usr]


# ---------------------------------------------------------------------------
# OpenAI orchestration
# ---------------------------------------------------------------------------

def classify_and_plan(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    trigger_1h: Optional[Dict[str, Any]] | None = None,
    model: Optional[str] | None = None,
    risk_mode: Optional[str] | None = None,
) -> Dict[str, Any]:
    mdl = model or DEFAULT_MODEL
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Phase 1: classify (VN)
    resp1 = client.chat.completions.create(
        model=mdl,
        messages=build_messages_classify(struct_4h, struct_1d, trigger_1h),
        temperature=0,
    )
    text1 = resp1.choices[0].message.content or ""
    ok1, cls, err1 = parse_json_block(text1)
    if not ok1:
        return {"ok": False, "error": f"classify_parse: {err1}", "raw": text1}

    action = (cls.get("action") or "").upper()
    side = (cls.get("side") or "").lower()
    if action != "ENTER" or side not in ("long", "short"):
        return {"ok": True, "decision": cls}

    # Phase 2: plan (VN, SL policy)
    resp2 = client.chat.completions.create(
        model=mdl,
        messages=build_messages_plan(
            struct_4h, struct_1d, side=side, classify_reasoning=cls, risk_mode=risk_mode or DEFAULT_RISK_MODE
        ),
        temperature=0,
    )
    text2 = resp2.choices[0].message.content or ""
    parsed = parse_plan_output(text2)
    if not parsed.get("ok"):
        return {"ok": False, "decision": cls, "error": parsed.get("error", "plan_parse_failed"), "raw": text2}

    plan = round_by_exchange(parsed["plan"])  # type: ignore[index]
    return {"ok": True, "decision": cls, "plan": plan}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_telegram_signal(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    trigger_1h: Optional[Dict[str, Any]] | None = None,
    model: Optional[str] | None = None,
) -> Dict[str, Any]:
    """Build signal for Telegram and an internal analysis text.

    Returns a dict with keys:
        ok: bool
        telegram_text: Optional[str]  # content to send to Telegram (no analysis)
        analysis_text: str            # detailed Vietnamese analysis for logs
        meta: { rr, eta, confidence, ... }
        decision: {...}
        plan: {...}
    """

    out = classify_and_plan(struct_4h, struct_1d, trigger_1h, model=model, risk_mode=DEFAULT_RISK_MODE)
    if not out.get("ok"):
        return out

    decision = out.get("decision")
    plan = out.get("plan")

    # If no entry → no telegram_text, but still provide a brief analysis block
    if not plan:
        analysis_text = (
            f"[ĐÁNH GIÁ] {decision.get('symbol', '?')} | {decision.get('action')} | "
            f"side={decision.get('side')} | conf={decision.get('confidence')}\n- "
            + "; ".join((decision.get("reasons") or [])[:4])
        )
        return {"ok": True, "decision": decision, "telegram_text": None, "analysis_text": analysis_text, "meta": {"note": "NO-ENTER"}}

    symbol: str = plan["symbol"]
    side: str = plan["side"]
    entries: List[float] = plan["entries"]
    stop: float = plan["stop"]
    tps: List[float] = plan["tps"]
    eta: Dict[str, Any] = plan.get("eta", {})

    # leverage by confidence
    conf = float(decision.get("confidence", 0.6) or 0.6)
    lev = pick_leverage(conf)

    # R:R
    rr = compute_rr(side, entries, stop, tps)

    # telegram_text (no "Nhận định")
    def _fmt(x: Any) -> str:
        try:
            return f"{float(x):.6f}".rstrip("0").rstrip(".")
        except Exception:  # noqa: BLE001
            return str(x)

    direction = "LONG" if side == "long" else "SHORT"
    lines: List[str] = [f"{symbol} | {direction}"]
    lines.append(f"Entry 1: {_fmt(entries[0])}")
    if len(entries) > 1:
        lines.append(f"Entry 2: {_fmt(entries[1])}")
    lines.append(f"SL: {_fmt(stop)}")
    for i in range(min(3, len(tps))):
        lines.append(f"TP{i + 1}: {_fmt(tps[i])}")
    lines.append(f"Đòn bẩy: {lev}")
    telegram_text = "\n".join(lines)

    # analysis_text (Vietnamese, data-driven from JSON)
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        msgs = build_messages_analysis(
            struct_4h,
            struct_1d,
            trigger_1h,
            decision,
            plan,
            rr_meta={"rr_min": rr["rr_min"], "rr_max": rr["rr_max"], "eta": eta},
        )
        respA = client.chat.completions.create(model=DEFAULT_MODEL, messages=msgs, temperature=0.2)
        analysis_text = (respA.choices[0].message.content or "").strip()
    except Exception:
        # Fallback concise analysis if GPT errors
        brief = "; ".join((decision.get("reasons") or [])[:3]) or "Xu hướng cùng pha; theo dõi phản ứng tại SR gần."
        analysis_text = (
            f"{symbol} | {direction}\n"
            f"- Lý do: {brief}\n"
            f"- R:R ~ {rr.get('rr_min')}→{rr.get('rr_max')} ; ETA: {eta if eta else '—'}"
        )

    meta = {"confidence": conf, "eta": eta, "rr": rr, "decision": decision, "plan": plan}
    return {
        "ok": True,
        "telegram_text": telegram_text,
        "analysis_text": analysis_text,
        "meta": meta,
        "decision": decision,
        "plan": plan,
    }
