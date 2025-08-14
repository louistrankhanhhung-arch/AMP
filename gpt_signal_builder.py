# gpt_signal_builder.py
from __future__ import annotations
import os, json, re, math
from typing import Any, Dict, List, Tuple, Optional

# ====== OpenAI client ======
from openai import OpenAI
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ====== JSON helpers ======
def parse_json_block(text: str) -> Tuple[bool, Any, str]:
    try:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.M)
        if m:
            return True, json.loads(m.group(1)), ""
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            return True, json.loads(text[start:end+1]), ""
        return False, None, "No JSON object found."
    except Exception as e:
        return False, None, f"JSON parse error: {e}"

def _trim_float(x: float, ndigits: int = 6) -> float:
    if x is None: return x
    return float(f"{float(x):.{ndigits}f}")

# ====== Schemas ======
CLASSIFY_SCHEMA = {
    "type":"object",
    "required":["symbol","side","action","confidence","reasons"],
    "properties":{
        "symbol":{"type":"string"},
        "side":{"type":"string","enum":["long","short","none"]},
        "action":{"type":"string","enum":["ENTER","WAIT","AVOID"]},
        "confidence":{"type":"number","minimum":0,"maximum":1},
        "reasons":{"type":"array","items":{"type":"string"},"maxItems":6},
        "trigger_hint":{"type":"object"}
    }
}

PLAN_SCHEMA = {
    "type":"object",
    "required":["symbol","side","entries","stop","tps","eta"],
    "properties":{
        "symbol":{"type":"string"},
        "side":{"type":"string","enum":["long","short"]},
        "entries":{"type":"array","items":{"type":"number"},"minItems":1,"maxItems":2},
        "stop":{"type":"number"},
        "tps":{"type":"array","items":{"type":"number"},"minItems":1,"maxItems":3},  # chỉ cần tới TP3
        "eta":{"type":"object"}  # {"tp1":"1-3 ngày",...}
    }
}

# ====== Prompts (JSON-only) ======
def build_messages_classify(struct_4h: Dict[str,Any], struct_1d: Dict[str,Any], trigger_1h: Optional[Dict[str,Any]]=None):
    ctx = {"struct_4h":struct_4h,"struct_1d":struct_1d,"trigger_1h":trigger_1h or {}}
    system = {
        "role":"system",
        "content":(
            "Bạn là trader kỹ thuật. Nhiệm vụ: PHÂN LOẠI nhanh side và hành động dựa trên JSON context (4H/1D + trigger_1h). "
            "Chỉ trả về JSON theo schema: " + json.dumps(CLASSIFY_SCHEMA, ensure_ascii=False) +
            "\nQuy tắc: 1D/4H đồng pha + 1H xác nhận → ENTER; mâu thuẫn/chưa rõ → WAIT; ngược mạnh → AVOID. "
            "reasons: 3–6 ý ngắn gọn, hạn chế liệt kê chỉ báo rườm rà."
        )
    }
    user = {"role":"user","content":[
        {"type":"text","text":"Context JSON:"},
        {"type":"text","text":json.dumps(ctx, ensure_ascii=False)}
    ]}
    return [system,user]

def build_messages_plan(struct_4h: Dict[str,Any], struct_1d: Dict[str,Any], side: str, classify_reasoning: Optional[Dict[str,Any]]=None):
    ctx = {"struct_4h":struct_4h,"struct_1d":struct_1d,"decision":{"side":side,"from_classify":classify_reasoning or {}}}
    system = {
        "role":"system",
        "content":(
            "Bạn là nhà giao dịch. Hãy lập kế hoạch vào lệnh ngắn gọn ở dạng JSON duy nhất theo schema: "
            + json.dumps(PLAN_SCHEMA, ensure_ascii=False) +
            "\nQuy tắc: long → stop < entries, TP tăng dần; short → stop > entries, TP giảm dần. "
            "Tối đa 2 Entry, tối đa 3 TP. Ưu tiên mức SR/BB/Fibo gần, SL dùng biên độ hợp lý. "
            "Không xuất HTML, chỉ JSON."
        )
    }
    user = {"role":"user","content":[
        {"type":"text","text":"Context JSON:"},
        {"type":"text","text":json.dumps(ctx, ensure_ascii=False)}
    ]}
    return [system,user]

# ====== Validate plan ======
def validate_plan(plan: Dict[str,Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    side = plan.get("side")
    entries = plan.get("entries") or []
    tps = plan.get("tps") or []
    stop = plan.get("stop")

    if side not in ("long","short"):
        errs.append("side must be long|short")
    if not isinstance(entries,list) or not entries:
        errs.append("entries must be 1-2 numbers")
    if not isinstance(tps,list) or not tps:
        errs.append("tps must be 1-3 numbers")
    if not isinstance(stop,(int,float)):
        errs.append("stop must be number")

    try:
        entries = [_trim_float(float(x)) for x in entries]
        tps     = [_trim_float(float(x)) for x in tps]
        stop    = _trim_float(float(stop))
    except Exception:
        errs.append("entries/stop/tps must be numeric")

    if not errs and side in ("long","short"):
        if side=="long":
            if not (stop < min(entries)): errs.append("For long: stop < min(entries)")
            if any(tps[i] >= tps[i+1] for i in range(len(tps)-1)): errs.append("For long: tps strictly increasing")
        else:
            if not (stop > max(entries)): errs.append("For short: stop > max(entries)")
            if any(tps[i] <= tps[i+1] for i in range(len(tps)-1)): errs.append("For short: tps strictly decreasing")
        if any(x<=0 for x in entries+[stop]+tps): errs.append("All price levels must be > 0")
    return (len(errs)==0), errs

def parse_plan_output(text: str) -> Dict[str,Any]:
    ok, obj, err = parse_json_block(text)
    if not ok:
        return {"ok":False,"error":err}
    valid, errs = validate_plan(obj)
    if not valid:
        return {"ok":False,"error":"; ".join(errs), "raw":obj}
    obj["entries"] = [_trim_float(x) for x in obj["entries"]]
    obj["stop"]    = _trim_float(obj["stop"])
    obj["tps"]     = [_trim_float(x) for x in obj["tps"]][:3]  # giữ tối đa TP3
    return {"ok":True,"plan":obj}

# ====== Compute leverage & R:R ======
def pick_leverage(conf: float) -> str:
    # map confidence -> đòn bẩy đề xuất
    if conf >= 0.80: return "x10"
    if conf >= 0.65: return "x5"
    return "x3"

def compute_rr(side: str, entries: List[float], stop: float, tps: List[float]) -> Dict[str,Any]:
    avg_entry = sum(entries)/len(entries)
    rr_list = []
    for tp in tps:
        if side=="long":
            risk = avg_entry - stop
            reward = tp - avg_entry
        else:
            risk = stop - avg_entry
            reward = avg_entry - tp
        if risk <= 0:
            rr = None
        else:
            rr = reward / risk
        rr_list.append(rr)
    return {
        "avg_entry": avg_entry,
        "rr_list": rr_list,
        "rr_min": min([x for x in rr_list if x is not None], default=None),
        "rr_max": max([x for x in rr_list if x is not None], default=None),
    }

# ====== Make comment (ngắn gọn, không lạm dụng chỉ báo) ======
def make_comment(symbol: str, side: str, decision: Dict[str,Any]) -> str:
    tips = {
        "long": "Ưu tiên vào theo xu hướng, chia 1–2 lệnh; đạt TP1 thì dời SL về hòa vốn, chốt dần tại TP2–TP3.",
        "short": "Ưu tiên bán theo xu hướng, vào từng phần; đạt TP1 dời SL về hòa vốn, chốt dần tại TP2–TP3."
    }
    # lấy 1–2 reason tiêu biểu
    reasons = [r for r in (decision.get("reasons") or []) if isinstance(r,str)]
    brief = "; ".join(reasons[:2]) if reasons else ""
    base = "Xu hướng đồng pha và có tín hiệu xác nhận." if not brief else brief
    return f"{base} {tips.get(side,'')}".strip()

# ====== Round by exchange dp (optional) ======
def round_by_exchange(plan: Dict[str,Any]) -> Dict[str,Any]:
    try:
        from universe import get_precisions_map, round_levels
        mp = get_precisions_map("KUCOIN", [plan["symbol"]])
        dp = mp[plan["symbol"]]["price_dp"]
        return round_levels(plan, dp)
    except Exception:
        return plan

# ====== OpenAI Orchestration ======
def classify_and_plan(struct_4h: dict, struct_1d: dict, trigger_1h: dict|None=None, model: str|None=None) -> dict:
    mdl = model or DEFAULT_MODEL
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Phase 1
    resp1 = client.chat.completions.create(model=mdl, messages=build_messages_classify(struct_4h, struct_1d, trigger_1h), temperature=0)
    text1 = resp1.choices[0].message.content or ""
    ok1, cls, err1 = parse_json_block(text1)
    if not ok1:
        return {"ok":False, "error":f"classify_parse: {err1}", "raw":text1}

    action = (cls.get("action") or "").upper()
    side   = (cls.get("side") or "").lower()
    if action != "ENTER" or side not in ("long","short"):
        return {"ok":True, "decision":cls}  # không vào lệnh → chỉ trả quyết định

    # Phase 2
    resp2 = client.chat.completions.create(model=mdl, messages=build_messages_plan(struct_4h, struct_1d, side=side, classify_reasoning=cls), temperature=0)
    text2 = resp2.choices[0].message.content or ""
    parsed = parse_plan_output(text2)
    if not parsed.get("ok"):
        return {"ok":False, "decision":cls, "error":parsed.get("error","plan_parse_failed"), "raw":text2}

    plan = round_by_exchange(parsed["plan"])
    return {"ok":True, "decision":cls, "plan":plan}

# ====== Telegram builder ======
def make_telegram_signal(struct_4h: dict, struct_1d: dict, trigger_1h: dict|None=None, model: str|None=None) -> dict:
    """
    Trả:
      {
        ok: bool,
        telegram_text: str  (để send thẳng cho bot),
        meta: { rr, eta, confidence, ... },
        decision: {...}, plan: {...}
      }
    """
    out = classify_and_plan(struct_4h, struct_1d, trigger_1h, model=model)
    if not out.get("ok"):
        return out

    decision = out.get("decision")
    plan     = out.get("plan")
    # không có plan (WAIT/AVOID) → trả decision
    if not plan:
        return {"ok":True, "decision":decision, "telegram_text":None, "meta":{"note":"NO-ENTER"}}

    symbol = plan["symbol"]
    side   = plan["side"]
    entries = plan["entries"]
    stop    = plan["stop"]
    tps     = plan["tps"]
    eta     = plan.get("eta", {})

    # leverage theo confidence
    conf = float(decision.get("confidence", 0.6) or 0.6)
    lev  = pick_leverage(conf)

    # R:R
    rr = compute_rr(side, entries, stop, tps)

    # format số
    def fmt(x): 
        try: return f"{float(x):.6f}".rstrip('0').rstrip('.')
        except: return str(x)

    # Direction text
    direction = "LONG" if side=="long" else "SHORT"

    # dựng message
    lines = [f"{symbol} | {direction}"]
    lines.append(f"Entry 1: {fmt(entries[0])}")
    if len(entries) > 1:
        lines.append(f"Entry 2: {fmt(entries[1])}")
    lines.append(f"SL: {fmt(stop)}")
    for i in range(min(3, len(tps))):
        lines.append(f"TP{i+1}: {fmt(tps[i])}")
    lines.append(f"Đòn bẩy: {lev}")

    comment = make_comment(symbol, side, decision)
    if comment:
        lines.append(f"Nhận định: {comment}")

    telegram_text = "\n".join(lines)

    meta = {
        "confidence": conf,
        "eta": eta,
        "rr": rr,
        "decision": decision,
        "plan": plan
    }
    return {"ok":True, "telegram_text": telegram_text, "meta": meta, "decision": decision, "plan": plan}
