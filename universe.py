"""
universe.py
-----------
Quản lý *universe* các mã theo dõi và chuẩn hoá *số thập phân* theo sàn (ccxt).

Tính năng chính:
- get_universe_from_env(): đọc SYMBOLS (CSV) từ ENV hoặc fallback list.
- get_precisions_map(exchange_id, symbols): trả về {symbol: {price_dp, amount_dp, min_qty, min_cost, tick_size, step_size}}.
- round_levels(plan, price_dp): làm tròn Entry/SL/TP theo số thập phân giá.
- CLI để xem nhanh bảng precisions.

Yêu cầu: ccxt
"""

from __future__ import annotations
import os, json, math, argparse
from typing import Dict, Any, List, Optional
import ccxt  # type: ignore

# ---------- helpers ----------

def _decimals_from_increment(value: Optional[float | int]) -> Optional[int]:
    """
    Từ tick size / step size -> số chữ số thập phân.
    - Nếu value là int (ví dụ 4) => hiểu như số chữ số thập phân cố định.
    - Nếu value là float (ví dụ 0.001) => dp = -log10(value)
    """
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    try:
        if value == 0:
            return None
        dp = int(round(-math.log10(float(value))))
        return max(0, dp)
    except Exception:
        return None

def _fmt(n: Optional[float | int], dp: int = 8) -> str:
    if n is None:
        return "-"
    try:
        s = f"{float(n):.{dp}f}"
        return s.rstrip('0').rstrip('.') if '.' in s else s
    except Exception:
        return str(n)

# ---------- universe ----------

DEFAULT_UNIVERSE = [
    "ADA/USDT","ARB/USDT","AVAX/USDT","BNB/USDT","BTC/USDT","DYDX/USDT","ETH/USDT","FET/USDT","INJ/USDT","LINK/USDT",
    "NEAR/USDT","PENDLE/USDT","SOL/USDT","SUI/USDT","ATOM/USDT","TRX/USDT","UNI/USDT","XRP/USDT",
]

def get_universe_from_env() -> List[str]:
    s = os.getenv("SYMBOLS","")
    if s.strip():
        return [x.strip() for x in s.split(",") if x.strip()]
    return DEFAULT_UNIVERSE[:]

def _make_exchange(exchange_id: str):
    ex_id = exchange_id.lower()
    if not hasattr(ccxt, ex_id):
        raise ValueError(f"Unsupported exchange: {exchange_id}")
    klass = getattr(ccxt, ex_id)
    ex = klass({
        "enableRateLimit": True,
        # có thể thêm apiKey/secret nếu cần private endpoints
    })
    return ex

def get_precisions_map(exchange_id: str, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Lấy thông tin precision/limits từ ccxt, trả map:
    {
      "SUI/USDT": {
         "price_dp": 4,
         "amount_dp": 2,
         "min_qty": 0.1,
         "min_cost": 10,
         "tick_size": 0.0001,
         "step_size": 0.01
      },
      ...
    }
    """
    ex = _make_exchange(exchange_id)
    markets = ex.load_markets()
    out: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        if sym not in markets:
            # không tìm thấy market này trên sàn -> bỏ qua
            continue
        m = markets[sym]

        # precision (nếu sàn trả về sẵn)
        p_price = None
        p_amt = None
        if isinstance(m.get("precision"), dict):
            p_price = m["precision"].get("price")
            p_amt   = m["precision"].get("amount")

        # limits (min)
        lims = m.get("limits") or {}
        min_qty  = (lims.get("amount") or {}).get("min")
        min_cost = (lims.get("cost") or {}).get("min")

        # increment/tick
        tick = m.get("priceIncrement") or m.get("tickSize") or (lims.get("price") or {}).get("min")
        step = m.get("amountIncrement") or (lims.get("amount") or {}).get("step")

        # suy ra dp nếu cần
        price_dp  = p_price if isinstance(p_price, int) else _decimals_from_increment(tick)
        amount_dp = p_amt   if isinstance(p_amt, int)   else _decimals_from_increment(step)

        out[sym] = {
            "symbol": sym,
            "price_dp": price_dp if price_dp is not None else 6,
            "amount_dp": amount_dp if amount_dp is not None else 4,
            "min_qty": min_qty,
            "min_cost": min_cost,
            "tick_size": tick,
            "step_size": step,
            "raw": {
                "precision": m.get("precision"),
                "limits": m.get("limits"),
                "info_keys": list((m.get("info") or {}).keys())
            }
        }
    return out

def round_levels(plan: Dict[str, Any], price_dp: int) -> Dict[str, Any]:
    """
    Làm tròn entries/stop/tps theo số chữ số thập phân của sàn.
    (dùng floor theo dp để tránh vượt step trên một số sàn)
    """
    q = 10 ** price_dp
    def r(x):
        try:
            return float(int(float(x) * q) / q)
        except Exception:
            return x
    plan = dict(plan)
    if isinstance(plan.get("entries"), list):
        plan["entries"] = [r(x) for x in plan["entries"]]
    if "stop" i
