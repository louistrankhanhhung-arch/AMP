# gpt_signal_builder.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI  # bắt buộc dùng OpenAI >= 1.x

# ========== Helpers ==========

def _fmt_price(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if x >= 100:
        return f"{x:.2f}"
    if x >= 10:
        return f"{x:.3f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"

def _pick(v, default=None):
    return default if v is None else v

def _ensure_candidate_from_args(first: Union[dict, str, None], **kwargs) -> Dict[str, Any]:
    """
    Hỗ trợ 2 cách gọi:
      - make_telegram_signal(candidate_dict)
      - make_telegram_signal(symbol="AVAX/USDT", side="LONG", entries=[...], sl=..., tps=[...], leverage="x5")
    """
    if isinstance(first, dict):
        return first
    c: Dict[str, Any] = {}
    if isinstance(first, str):
        c["symbol"] = first
    c["symbol"]   = _pick(c.get("symbol"), kwargs.get("symbol"))
    c["side"]     = kwargs.get("side") or kwargs.get("direction")
    c["entries"]  = kwargs.get("entries") or kwargs.get("entry") or kwargs.get("entry_prices")
    if c["entries"] is None:
        e1 = kwargs.get("entry1"); e2 = kwargs.get("entry2")
        c["entries"] = [e for e in (e1, e2) if e is not None]
    c["sl"]       = kwargs.get("sl") or kwargs.get("stop") or kwargs.get("stop_loss")
    c["tps"]      = kwargs.get("tps") or [kwargs.get("tp1"), kwargs.get("tp2"), kwargs.get("tp3")]
    c["leverage"] = kwargs.get("leverage") or kwargs.get("lev")
    c["meta"]     = kwargs.get("meta") or {}
    return c

def _require_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)

# ========== Public API ==========

def make_telegram_signal(candidate_or_symbol: Union[Dict[str, Any], str], **kwargs) -> str:
    """
    Tạo text gửi Telegram (KHÔNG có 'Nhận định').
    """
    c = _ensure_candidate_from_args(candidate_or_symbol, **kwargs)

    sym   = c.get("symbol", "UNKNOWN")
    side  = (c.get("side") or "WAIT").upper()
    ent   = c.get("entries") or []
    sl    = c.get("sl")
    tps   = [tp for tp in (c.get("tps") or []) if tp is not None]
    lev   = c.get("leverage")

    lines: List[str] = []
    lines.append(f"{sym} | {side}")
    if len(ent) >= 1: lines.append(f"Entry 1: {_fmt_price(ent[0])}")
    if len(ent) >= 2: lines.append(f"Entry 2: {_fmt_price(ent[1])}")
    if sl is not None: lines.append(f"SL: {_fmt_price(sl)}")
    if len(tps) >= 1: lines.append(f"TP1: {_fmt_price(tps[0])}")
    if len(tps) >= 2: lines.append(f"TP2: {_fmt_price(tps[1])}")
    if len(tps) >= 3: lines.append(f"TP3: {_fmt_price(tps[2])}")
    if lev:
        lev_str = str(lev)
        if isinstance(lev, (int, float)):
            lev_str = f"x{int(lev)}"
        elif not lev_str.startswith("x"):
            lev_str = f"x{lev_str}"
        lines.append(f"Đòn bẩy: {lev_str}")
    return "\n".join(lines)


def make_analysis_log(
    symbol: str,
    *,
    structs: Dict[str, Any],
    decision: Dict[str, Any],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Sinh 'Nhận định' THUẦN VIỆT bắt buộc qua OpenAI — để LƯU LOG nội bộ (không gửi Telegram).
    - symbol: "AVAX/USDT"
    - structs: dict chứa 1H/4H/1D (đã build từ engine; snapshot/trend/events/context_levels...)
    - decision: dict signal (side, entries, sl, tps, rr, eta...)
    """
    try:
        client = _require_openai_client()
    except Exception as e:
        return f"[gpt-error] {e}"

    mdl = model or os.getenv("OPENAI_MODEL") or "gpt-4o"
    temp = float(os.getenv("OPENAI_TEMPERATURE", "0.2")) if temperature is None else float(temperature)

    # Rút gọn dữ liệu gửi GPT (đủ ý, tránh thừa token)
    def _core(s: Dict[str, Any]) -> Dict[str, Any]:
        if not s: return {}
        snap = s.get("snapshot", {})
        return {
            "timeframe": s.get("timeframe"),
            "close": snap.get("close"),
            "ema20": snap.get("ema20"),
            "ema50": snap.get("ema50"),
            "bb_up": snap.get("bb_up"),
            "bb_low": snap.get("bb_low"),
            "rsi": snap.get("rsi"),
            "atr": snap.get("atr"),
            "trend": s.get("trend"),
            "events": s.get("events"),
            "context_levels": s.get("context_levels"),
            "market_structure": s.get("market_structure"),
            "stats": s.get("stats"),
        }

    payload = {
        "symbol": symbol,
        "tf_1H": _core(structs.get("1H", {})),
        "tf_4H": _core(structs.get("4H", {})),
        "tf_1D": _core(structs.get("1D", {})),
        "decision": {
            "side": (decision or {}).get("side"),
            "entries": (decision or {}).get("entries"),
            "sl": (decision or {}).get("sl"),
            "tps": (decision or {}).get("tps"),
            "rr_min": (decision or {}).get("rr_min"),
            "rr_max": (decision or {}).get("rr_max"),
            "eta": (decision or {}).get("eta"),
        }
    }

    system = (
        "Bạn là trader crypto viết nhận định **thuần Việt** gọn, rõ, có hành động. "
        "Chỉ tạo văn bản, không markdown nặng, không lẫn tiếng Anh. "
        "Trọng tâm: (1) xu hướng 1D/4H, (2) SR quan trọng và vùng mỏng, "
        "(3) động lượng RSI/MA20, (4) có/không breakout-breakdown đã xác nhận, "
        "(5) kế hoạch hành động: vào/thoát/dời SL, (6) nêu RR & ETA (nếu có). "
        "Tối đa 8–10 dòng, tránh lặp lại khuôn mẫu."
    )
    user = (
        "Phân tích dữ liệu sau và viết nhận định nội bộ (không gửi Telegram):\n"
        f"{payload}\n"
        "Yêu cầu thêm: tránh giáo điều, tập trung điều kiện kích hoạt/huỷ kèo, "
        "nêu rõ mốc giá theo dữ liệu, không liệt kê chỉ báo thừa."
    )

    try:
        resp = client.chat.completions.create(
            model=mdl,
            temperature=temp,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return "[gpt-error] Empty content"
        return content
    except Exception as e:
        return f"[gpt-error] {e}"
