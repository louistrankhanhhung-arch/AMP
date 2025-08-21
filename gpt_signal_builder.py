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

# ====== Schema output mong đợi từ GPT (2 kế hoạch) ======
CLASSIFY_SCHEMA = {
    "symbol": "BTC/USDT",
    "plans": {
        "intraday_1h": {
            "decision": "ENTER | WAIT | AVOID",
            "side": "long | short",
            "confidence": 0.0,
            "strategy": "retest | reclaim | breakout | range | countertrend",
            "entries": [0.0],
            "sl": 0.0,
            "tps": [0.0, 0.0],
            "reasons": ["..."],
            "trigger_hint": "nếu WAIT: nêu điều kiện kích hoạt cụ thể"
        },
        "swing_4h": {
            "decision": "ENTER | WAIT | AVOID",
            "side": "long | short",
            "confidence": 0.0,
            "strategy": "trend-follow | breakout | retest | range",
            "entries": [0.0],
            "sl": 0.0,
            "tps": [0.0, 0.0],
            "reasons": ["..."],
            "trigger_hint": "nếu WAIT: nêu điều kiện kích hoạt cụ thể"
        }
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
    Format Telegram *chỉ dùng khi ENTER*:
    {Direction} | {Mã} (LABEL)
    Leverage:
    Entry:
    SL:
    TP:
    """
    side = (decision.get("side") or "long").lower()
    direction = "Long" if side == "long" else "Short"
    entries = decision.get("entries") or decision.get("entry") or []
    tps = decision.get("tps") or decision.get("tp") or []
    sl = decision.get("sl")
    leverage = decision.get("leverage")

    hdr = f"{direction} | {symbol.replace('/', '')}"
    if label:
        hdr += f" ({label})"

    return "\n".join([
        hdr,
        f"Leverage: {leverage if leverage is not None else '-'}",
        f"Entry: {_fmt_list(entries)}",
        f"SL: {('-' if sl is None else (f'{float(sl):.6f}' if isinstance(sl,(int,float,str)) else str(sl)))}",
        f"TP: {_fmt_list(tps)}",
    ])

def _analysis_lines(symbol: str, decision: Dict[str, Any], tag: str) -> List[str]:
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

def _merge_analysis(symbol: str, p1: Optional[Dict[str, Any]], p2: Optional[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    if p1:
        blocks.append("\n".join(_analysis_lines(symbol, p1, "INTRADAY")))
    if p2:
        blocks.append("\n".join(_analysis_lines(symbol, p2, "SWING")))
    if not blocks:
        return f"[ĐÁNH GIÁ] {symbol} | N/A"
    return "\n\n".join(blocks)

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
    Gửi đầy đủ context 1D/4H/1H. Yêu cầu GPT trả về 2 kế hoạch song song.
    """

    # Nếu nhận vào là DataFrame thì chuyển sang struct JSON gọn
    s4 = _df_to_struct(struct_4h, "4H") if hasattr(struct_4h, "to_dict") else (struct_4h or {})
    s1 = _df_to_struct(struct_1d, "1D") if hasattr(struct_1d, "to_dict") else (struct_1d or {})
    sH = _df_to_struct(trigger_1h, "1H") if (trigger_1h is not None and hasattr(trigger_1h, "to_dict")) else (trigger_1h or {})

    ctx = {"struct_4h": s4, "struct_1d": s1, "struct_1h": sH, "symbol": symbol_hint,}

    # === DEBUG: in & ghi JSON đầu vào GPT khi DEBUG_GPT_INPUT=1 ===
    try:
        sym_for_dump = (
            symbol_hint
            or _safe(struct_4h, "symbol")
            or _safe(struct_1d, "symbol")
            or _safe(trigger_1h, "symbol")
            or "SYMBOL"
        )
        debug_print_gpt_input(ctx)                  # in ra log
        debug_dump_gpt_input(sym_for_dump, ctx, tag="ctx")  # ghi file (nếu bật)
    except Exception:
        pass

    system = {
        "role": "system",
        "content": (
            "Bạn là trader kỹ thuật. Với JSON 3 khung **1D / 4H / 1H**, hãy trả về **2 kế hoạch độc lập**:\n"
            "1) intraday_1h: setup theo 1H (lướt sóng ngắn).\n"
            "2) swing_4h: setup theo 4H (đồng pha 1D, giữ lệnh dài hơn).\n\n"
            "Tiêu chuẩn:\n"
            "- Intraday (1H): chấp nhận RSI quá mua/bán nếu có xác nhận volume; yêu cầu R:R ≥ 1.5; ưu tiên reclaim/retest/mini-breakout; SL chặt; vị thế ≤ 0.3–0.5R.\n"
            "- Swing (4H): 4H phải đồng pha với 1D; tránh trade ngược xu hướng lớn; yêu cầu R:R ≥ 2.0; vị thế 1.0R chuẩn.\n"
            "Nếu WAIT: thêm trigger_hint (điểm hoặc kịch bản cụ thể). Nếu AVOID: ghi lý do ngắn gọn.\n"
            "Trả về JSON đúng theo schema sau (tiếng Việt, KHÔNG thêm văn bản ngoài JSON):\n"
            + json.dumps(CLASSIFY_SCHEMA, ensure_ascii=False)
        ),
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Context JSON (1D/4H/1H):"},
            {"type": "text", "text": json.dumps(ctx, ensure_ascii=False)},
        ],
    }
    return [system, user]

# ====== Hàm chính ======
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
    - Nhận 2 kế hoạch: intraday_1h & swing_4h.
    - Nếu bất kỳ kế hoạch nào ENTER: tạo telegram_text tương ứng (kèm nhãn).
      Để tương thích ngược: chọn intraday nếu ENTER, nếu không chọn swing; đồng thời trả thêm telegram_texts (list).
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

        plans = data.get("plans") or {}
        p_intra = plans.get("intraday_1h") or {}
        p_swing = plans.get("swing_4h") or {}

        # Chuẩn hoá fields cho từng plan
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

        p_intra = _norm(p_intra)
        p_swing = _norm(p_swing)

        # Build telegram texts (nếu ENTER)
        telegram_texts: List[str] = []
        enter_plans: List[Dict[str, Any]] = []

        if p_intra["decision"] == "ENTER":
            telegram_texts.append(_render_simple_signal(symbol, p_intra, label="INTRADAY"))
            enter_plans.append({"label": "INTRADAY", **p_intra})

        if p_swing["decision"] == "ENTER":
            telegram_texts.append(_render_simple_signal(symbol, p_swing, label="SWING"))
            enter_plans.append({"label": "SWING", **p_swing})

        # Tương thích ngược: chọn 1 plan “chính” nếu có
        telegram_text = None
        plan_primary = None
        if telegram_texts:
            # ưu tiên intraday; nếu không có thì swing
            if p_intra["decision"] == "ENTER":
                telegram_text = telegram_texts[0]  # intraday đứng trước nếu có
                plan_primary = {
                    "signal_id": f"{symbol.replace('/','')}-{int(time.time())}",
                    "timeframe": "1H",
                    "side": p_intra["side"],
                    "strategy": p_intra.get("strategy") or "GPT-plan",
                    "entries": p_intra.get("entries") or [],
                    "sl": p_intra.get("sl"),
                    "tps": p_intra.get("tps") or [],
                    "leverage": p_intra.get("leverage"),
                    "eta": p_intra.get("eta"),
                }
            else:
                telegram_text = telegram_texts[-1]  # swing
                plan_primary = {
                    "signal_id": f"{symbol.replace('/','')}-{int(time.time())}",
                    "timeframe": "4H",
                    "side": p_swing["side"],
                    "strategy": p_swing.get("strategy") or "GPT-plan",
                    "entries": p_swing.get("entries") or [],
                    "sl": p_swing.get("sl"),
                    "tps": p_swing.get("tps") or [],
                    "leverage": p_swing.get("leverage"),
                    "eta": p_swing.get("eta"),
                }

        # Phần phân tích (gộp 2 block)
        analysis_text = _merge_analysis(symbol, p_intra, p_swing)

        return {
            "ok": True,
            "symbol": symbol,
            # tương thích ngược:
            "telegram_text": telegram_text,           # 1 message chính (nếu có)
            "analysis_text": analysis_text,           # luôn có
            "plan": plan_primary,                     # 1 plan chính (nếu có)
            # mở rộng:
            "telegram_texts": telegram_texts,         # tất cả ENTER
            "plans": {                                # trả full 2 plan
                "intraday_1h": p_intra,
                "swing_4h": p_swing
            },
            "enter_plans": enter_plans,               # các plan ENTER có nhãn
            "meta": {
                "intraday_decision": p_intra["decision"],
                "swing_decision": p_swing["decision"],
                "intraday_conf": p_intra.get("confidence"),
                "swing_conf": p_swing.get("confidence"),
            },
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
