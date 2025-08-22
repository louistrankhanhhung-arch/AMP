# gpt_signal_builder.py
import os
import json
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from openai import OpenAI

def _df_ok(df):
    return (df is not None) and (not getattr(df, "empty", False))

def _last(df, col):
    # lấy giá trị cuối cùng an toàn để tránh so sánh Series
    return float(df[col].iloc[-1])


# ==== Debug helpers (standalone fallback) ====
# Mức debug:
#   DEBUG_GPT_INPUT=0  -> tắt hết
#   DEBUG_GPT_INPUT=1  -> in SLIM (không raw rows) + ghi file đầy đủ
#   DEBUG_GPT_INPUT=2  -> in FULL (có rows) + ghi file đầy đủ
def _debug_level() -> int:
    try:
        return int(os.getenv("DEBUG_GPT_INPUT", "0") or "0")
    except Exception:
        return 0

def _debug_enabled() -> bool:
    return _debug_level() > 0

def _ctx_slim(ctx: Dict[str, Any], preview_rows: int = 0) -> Dict[str, Any]:
    # lọc bỏ raw rows; chỉ hiển thị khung TF, số dòng, và (tuỳ chọn) vài dòng cuối
    def slim_one(s):
        if not isinstance(s, dict):
            return {}
        rows = s.get("rows") or []
        out = {
            "timeframe": s.get("timeframe"),
            "rows_count": len(rows),
        }
        if preview_rows and isinstance(rows, list):
            tail = rows[-preview_rows:]
            out["rows_preview"] = tail
            # thêm danh sách cột để tham khảo nhanh
            if tail:
                out["cols"] = list(tail[-1].keys())
        return out
    return {
        "struct_4h": slim_one(ctx.get("struct_4h", {})),
        "struct_1d": slim_one(ctx.get("struct_1d", {})),
        "struct_1h": slim_one(ctx.get("struct_1h", {})),
    }

def debug_print_gpt_input(ctx: Dict[str, Any]) -> None:
    lvl = _debug_level()
    if lvl <= 0:
        return
    # SLIM khi lvl=1, FULL khi lvl>=2
    if lvl == 1:
        preview = int(os.getenv("DEBUG_CTX_PREVIEW", "0") or "0")  # mặc định 0 = không in rows
        obj = _ctx_slim(ctx, preview_rows=preview)
        txt = json.dumps(obj, ensure_ascii=False, indent=2)
        print("[DEBUG GPT INPUT] context (slim):\n", txt)
    else:
        txt = json.dumps(ctx, ensure_ascii=False, indent=2)
        print("[DEBUG GPT INPUT] context (full):\n", txt)

def debug_dump_gpt_input(symbol: str, ctx: Dict[str, Any], tag: str = "ctx") -> None:
    if not _debug_enabled():
        return
    try:
        out_dir = os.getenv("DEBUG_GPT_DIR", "/mnt/data/gpt_inputs")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(out_dir, f"{symbol or 'UNK'}_{tag}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ctx, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] saved GPT input -> {path}")
    except Exception:
        pass


# ====== Config ======
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # dùng gpt-4o theo yêu cầu
client = OpenAI()

# ====== Schema output mong đợi từ GPT (1 kế hoạch T+) ======
CLASSIFY_SCHEMA = {
    "symbol": "BTC/USDT",
    "tplus": {
        "decision": "ENTER | WAIT | AVOID",
        "side": "long | short",
        "confidence": 0.0,
        "strategy": "trend-follow | breakout | reclaim | retest | range | countertrend",
        "entries": [0.0],
        "sl": 0.0,
        "tps": [0.0, 0.0],
        "eta": "thời gian ước tính đạt TP1/TP2 (ví dụ: '4–12h')",
        "leverage": "tùy chọn, nếu có",
        "reasons": ["..."],
        "trigger_hint": "nếu WAIT: nêu điều kiện kích hoạt cụ thể"
    }
}

# ====== Helpers ======
def _safe(d: Optional[Dict], *keys, default=None):
    # Không dùng `d or {}` vì DataFrame không có truth value
    cur = d if d is not None else {}
    for k in keys:
        if cur is None:
            return default
        # Chỉ đi tiếp nếu là dict-like
        if hasattr(cur, "get"):
            cur = cur.get(k)
        else:
            return default
    return default if cur is None else cur


def _parse_json_from_text(txt: str) -> Dict[str, Any]:
    """
    Cố gắng tách JSON từ nội dung GPT trả về (hỗ trợ có/không code fence).
    """
    t = (txt or "").strip()

    # code fence
    if t.startswith("```"):
        # loại bỏ fence, cả '```json'
        t = t.strip().strip("`")
        if t.lower().startswith("json"):
            p = t.find("{")
            if p != -1:
                t = t[p:]

    # tìm block {...} lớn nhất
    try:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start:end + 1])
    except Exception:
        pass

    # fallback: thử parse trực tiếp
    try:
        return json.loads(t)
    except Exception:
        return {}

def _fmt_list(nums):
    if not nums:
        return "-"
    try:
        return ", ".join(f"{float(x):.6f}" for x in nums)
    except Exception:
        return ", ".join(str(x) for x in nums)


def _render_simple_signal(symbol: str, decision: Dict[str, Any], label: str | None = None) -> str:
    """
    Format Telegram *chỉ dùng khi ENTER* (đơn giản, không [signal], không Leverage):
    {DIRECTION} | {SYMBOL}
    {LABEL}

    Entry: ...
    Stop: ...
    TP: ...
    
    Strategy: ...
    """
    side = (decision.get("side") or "long").lower()
    direction = "LONG" if side == "long" else "SHORT"
    # giữ nguyên symbol dạng có "/" nếu đã có; nếu không, thử chèn "/USDT" khi phù hợp
    sym = symbol if "/" in symbol else (symbol[:-4] + "/USDT" if symbol.endswith("USDT") and len(symbol) > 4 else symbol)

    entries = decision.get("entries") or decision.get("entry") or []
    tps = decision.get("tps") or decision.get("tp") or []
    sl = decision.get("sl")
    strategy = decision.get("strategy") or "-"

    def _fmt_list_simple(nums):
        if not nums:
            return "-"
        try:
            return ", ".join(f"{float(x):g}" for x in nums)
        except Exception:
            return ", ".join(str(x) for x in nums)

    lines = [f"{direction} | {sym}"]
    if label:
        lines.append(str(label).upper())
    lines.append("")
    lines.append(f"Entry: {_fmt_list_simple(entries)}")
    lines.append(f"Stop: {('-' if sl is None else (f'{float(sl):g}' if isinstance(sl,(int,float,str)) else str(sl)))}")
    lines.append(f"TP: {_fmt_list_simple(tps)}")
    lines.append("")
    lines.append(f"Strategy: {strategy}")
    return "\n".join(lines)

def _analysis_lines(symbol: str, decision: Dict[str, Any], tag: str = "T+") -> List[str]:
    act = (decision.get("decision") or decision.get("action") or "").upper()
    side = (decision.get("side") or "long").lower()
    conf = decision.get("confidence")
    reasons = decision.get("reasons") or []
    hint = decision.get("trigger_hint")

    lines = [f"[ĐÁNH GIÁ {tag}] {symbol} | {act} | side={side} | conf={conf}"]
    for r in reasons[:8]:
        if isinstance(r, str) and r.strip():
            lines.append(f"- {r.strip()}")
    if act == "WAIT" and hint:
        lines.append(f"- Trigger: {hint}")
    return lines

def _analysis_one(symbol: str, plan: Optional[Dict[str, Any]]) -> str:
    if not plan:
        return f"[ĐÁNH GIÁ] {symbol} | N/A"
    return "\n".join(_analysis_lines(symbol, plan, "T+"))

# ====== Prompt xây dựng ======
def _df_to_struct(df, tf_label: str) -> Dict[str, Any]:
    # cắt gọn để gửi GPT; giữ 200 dòng cuối, reset index
    try:
        df2 = df.tail(200).reset_index(drop=True)
        # ưu tiên các cột phổ biến; nếu thiếu thì cứ để nguyên cột hiện có
        prefer = [c for c in ["time","open","high","low","close","volume",
                              "ema20","ema50","ema200","rsi","vwap"] if c in df2.columns]
        cols = prefer if prefer else list(df2.columns)
        return {
            "timeframe": tf_label,
            "rows": df2[cols].to_dict(orient="records"),
        }
    except Exception:
        # fallback an toàn
        return {"timeframe": tf_label, "rows": []}

def build_messages_classify(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    trigger_1h: Optional[Dict[str, Any]] = None,
    symbol_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Gửi đầy đủ context 1D/4H/1H. Yêu cầu GPT trả về 1 kế hoạch T+.
    ENTRY PROXIMITY GATE: tol = 0.015 * current_price.
    Bảo đảm tương thích /v1/chat/completions: messages là LIST và content là STRING.
    """
    # Helper an toàn cho json.dumps (tránh lỗi numpy/timestamp)
    def _safe_dumps_local(obj):
        try:
            return json.dumps(obj, ensure_ascii=False, allow_nan=False, default=str)
        except Exception:
            return json.dumps(str(obj), ensure_ascii=False)

    # Nếu nhận vào là DataFrame thì chuyển sang struct JSON gọn
    s4 = _df_to_struct(struct_4h, "4H") if hasattr(struct_4h, "to_dict") else (struct_4h or {})
    s1 = _df_to_struct(struct_1d, "1D") if hasattr(struct_1d, "to_dict") else (struct_1d or {})
    sH = _df_to_struct(trigger_1h, "1H") if (trigger_1h is not None and hasattr(trigger_1h, "to_dict")) else (trigger_1h or {})

    ctx = {"struct_4h": s4, "struct_1d": s1, "struct_1h": sH, "symbol": symbol_hint}

    # === DEBUG: in & ghi JSON đầu vào GPT khi DEBUG_GPT_INPUT=1 ===
    try:
        sym_for_dump = (
            symbol_hint
            or _safe(struct_4h, "symbol")
            or _safe(struct_1d, "symbol")
            or _safe(trigger_1h, "symbol")
            or "SYMBOL"
        )
        debug_print_gpt_input(ctx)                   # in ra log
        debug_dump_gpt_input(sym_for_dump, ctx, tag="ctx")  # ghi file (nếu bật)
    except Exception:
        pass

    # ====== SYSTEM TEXT for T+ (gộp 1H-4H-1D) with ENTRY PROXIMITY GATE ======
    # Lấy schema text an toàn
    try:
        schema_text = _safe_dumps_local(CLASSIFY_SCHEMA)
    except Exception:
        schema_text = _safe_dumps_local({"note": "CLASSIFY_SCHEMA unavailable"})

    system_text = (
        "[ENTRY PROXIMITY GATE — BẮT BUỘC]\n"
        "- current_price = giá đóng nến gần nhất của khung 1H trong dữ liệu đầu vào.\n"
        "- entries = mảng mức vào lệnh đề xuất. Định nghĩa vùng: lo = min(entries), hi = max(entries). Nếu chỉ có 1 mức thì coi lo = hi.\n"
        "- tolerance (nới): tol = 0.015 * current_price. Không sử dụng ATR_1H trong ENTRY PROXIMITY GATE.\n"
        "- Áp dụng CHO KẾ HOẠCH T+ DUY NHẤT:\n"
        "    * Chỉ được trả action = \"ENTER\" nếu current_price ∈ [lo - tol, hi + tol]. (tức giá đang ở ngay/sát vùng entry, có thể vào ngay).\n"
        "    * Nếu nằm ngoài khoảng trên ⇒ action = \"WAIT\" và BẮT BUỘC có trigger_hint (điểm/kịch bản cụ thể, ví dụ: 'retest biên lo–hi ± tol' hoặc 'close 1H > ngưỡng breakout + volume > SMA20').\n"
        "    * Không đuổi giá: nếu Long và current_price > hi + tol (hoặc Short và current_price < lo - tol) ⇒ WAIT.\n"
        "    * Với strategy \"breakout\"/\"reclaim\": trigger_hint = điều kiện đóng nến vượt biên + xác nhận volume.\n"
        "      Với \"retest\"/\"pullback\": trigger_hint = chạm lại zone lo–hi (có thể ghi biên ±tol).\n"
        "    * Nếu entries rỗng/không hợp lệ ⇒ luôn WAIT và nêu rõ trigger_hint.\n"
        "- Output chỉ JSON đúng schema; KHÔNG thêm văn bản ngoài JSON.\n"
        "\n"
        "Bạn là trader kỹ thuật. Với JSON 3 khung **1D / 4H / 1H**, hãy trả về **1 kế hoạch T+ duy nhất** (giữ tối đa 2–5 ngày, có thể vào ngay khi đủ điều kiện proximity):\n"
        "- Ưu tiên 4H đồng pha 1D; nếu countertrend (4H ngược 1D) phải ghi rõ 'countertrend' trong strategy và hạ confidence.\n"
        "- Yêu cầu tối thiểu: R:R kỳ vọng ≥ 1.5; SL/TP rõ ràng; có **ETA** dự kiến đạt TP1/TP2.\n"
        "- Nếu WAIT: thêm trigger_hint (điểm hoặc kịch bản cụ thể). Nếu AVOID: ghi lý do ngắn gọn.\n"
        "Trả về JSON đúng theo schema sau (tiếng Việt, KHÔNG thêm văn bản ngoài JSON):\n"
        f"{schema_text}"
    )

    system = {"role": "system", "content": system_text}

    # Với /v1/chat/completions, content của user nên là STRING (không phải list part)
    user_text = "Context JSON (1D/4H/1H):\n" + _safe_dumps_local(ctx)
    user = {"role": "user", "content": user_text}

    messages = [system, user]

    # Guard cuối: đảm bảo đúng định dạng
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        raise ValueError("build_messages_classify produced invalid messages")
    if not isinstance(system.get("content"), str) or not isinstance(user.get("content"), str):
        raise ValueError("message content must be string for /v1/chat/completions")

    return messages

def make_telegram_signal(
    s4h: Any,
    s1d: Any,
    trigger_1h: Optional[Any] = None,
    symbol_hint: Optional[str] = None,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    # --- Back-compat aliases: phần thân vẫn dùng struct_* ---
    struct_4h = s4h
    struct_1d = s1d
    struct_1h = trigger_1h

    # Guard: tránh lỗi truth value với DataFrame
    if not (_df_ok(struct_4h) and _df_ok(struct_1d) and (struct_1h is None or _df_ok(struct_1h))):
        return {"ok": False, "error": "missing/empty frame(s)"}
    """
    - Gọi GPT-4o với 1H/4H/1D đầy đủ.
    - Nhận 1 kế hoạch: T+.
    - Nếu ENTER: tạo telegram_text tương ứng.
    - WAIT/AVOID: chỉ log analysis (gồm trigger_hint nếu WAIT).
    """
    try:
        msgs = build_messages_classify(struct_4h, struct_1d, trigger_1h=trigger_1h, symbol_hint=symbol_hint)
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=msgs,
            temperature=0.0,
            max_tokens=2200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _parse_json_from_text(raw)

        if not isinstance(data, dict) or not data:
            return {"ok": False, "error": "GPT không trả JSON hợp lệ", "raw": raw}

        symbol = (symbol_hint or data.get("symbol") or "SYMBOL")

        p_tplus = data.get("tplus") or {}

        # Chuẩn hoá fields cho plan
        def _norm(p: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "decision": (p.get("decision") or p.get("action") or "WAIT").upper(),
                "side": (p.get("side") or "long").lower(),
                "confidence": p.get("confidence"),
                "strategy": p.get("strategy"),
                "entries": p.get("entries") or p.get("entry") or [],
                "sl": p.get("sl"),
                "tps": p.get("tps") or p.get("tp") or [],
                "reasons": p.get("reasons") or (p.get("analysis") and [p["analysis"]] or []),
                "trigger_hint": p.get("trigger_hint"),
                "leverage": p.get("leverage"),
                "eta": p.get("eta"),
            }

        p_tplus = _norm(p_tplus)

        # Build telegram text (nếu ENTER)
        telegram_texts: List[str] = []
        enter_plans: List[Dict[str, Any]] = []

        if p_tplus["decision"] == "ENTER":
            telegram_texts.append(_render_simple_signal(symbol, p_tplus, label="T+"))
            enter_plans.append({"label": "T+", **p_tplus})

        # Chọn plan “chính” nếu có
        telegram_text = None
        plan_primary = None
        if telegram_texts:
            telegram_text = telegram_texts[0]
            plan_primary = {
                "signal_id": f"{symbol.replace('/','')}-{int(time.time())}",
                "timeframe": "T+",
                "side": p_tplus["side"],
                "strategy": p_tplus.get("strategy") or "GPT-plan",
                "entries": p_tplus.get("entries") or [],
                "sl": p_tplus.get("sl"),
                "tps": p_tplus.get("tps") or [],
                "leverage": p_tplus.get("leverage"),
                "eta": p_tplus.get("eta"),
            }

        # Phần phân tích (1 block T+)
        analysis_text = _analysis_one(symbol, p_tplus)

        return {
            "ok": True,
            "symbol": symbol,
            # tương thích ngược:
            "telegram_text": telegram_text,           # 1 message chính (nếu có)
            "analysis_text": analysis_text,           # luôn có
            "plan": plan_primary,                     # 1 plan chính (nếu có)
            # mở rộng:
            "telegram_texts": telegram_texts,         # 1 phần tử nếu ENTER
            "plans": {"tplus": p_tplus},              # trả full T+ plan
            "enter_plans": enter_plans,               # nếu ENTER
            "meta": {
                "tplus_decision": p_tplus["decision"],
                "tplus_conf": p_tplus.get("confidence"),
            },
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
