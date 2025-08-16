# -*- coding: utf-8 -*-
"""
performance_logger.py — ghi nhận và tổng hợp hiệu suất giao dịch.

• Ghi log mỗi kèo khi đóng (TP-all / SL / đóng tay): log_closed_signal(payload)
• Tổng hợp KPI theo kỳ: summarize(period), với period ∈ {"7d", "30d", "all"}
• Phân rã KPI: overall, theo từng mã (symbol), theo side (long/short), theo strategy,
  và kết hợp symbol×side, strategy×side.
• Có hàm summarize_print(period) để in ra logs gọn gàng.

Đầu vào (payload) kỳ vọng tối thiểu:
{
  "signal_id": str,
  "symbol": "SUI/USDT",
  "side": "long"|"short",
  "opened_at": ISO8601,
  "closed_at": ISO8601,
  "result": "TP"|"SL"|"CLOSED",
  "realized_R": float,
  ... (tùy chọn: entries, stop, tps, strategy, rr_theoretical, ...)
}

Thư mục log mặc định: /mnt/data/perf_logs  (tạo tự động)
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

# =========================
# Cấu hình & tiện ích chung
# =========================
LOG_DIR = os.getenv("PERF_LOG_DIR", "/mnt/data/perf_logs")
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

ISO_Z = "%Y-%m-%dT%H:%M:%S"

def _iso_now_utc() -> str:
  return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"

def _parse_dt(ts: Optional[str]) -> Optional[datetime]:
  if not ts:
    return None
  try:
    # Hỗ trợ dạng có 'Z' hoặc không
    t = ts.rstrip("Z")
    # Cho phép microseconds
    return datetime.fromisoformat(t)
  except Exception:
    return None

# =========================
# 1) Ghi log khi kèo đóng
# =========================

def log_closed_signal(payload: Dict[str, Any]) -> None:
  """Ghi log 1 kèo đã đóng thành file JSON: closed_<signal_id>.json"""
  sig = payload.get("signal_id") or f"noid_{int(datetime.now().timestamp())}"
  path = Path(LOG_DIR) / f"closed_{sig}.json"
  payload = dict(payload)  # copy
  payload.setdefault("logged_at", _iso_now_utc())
  with path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

# =========================
# 2) Đọc & lọc log theo kỳ
# =========================

def _iter_closed(start: Optional[datetime] = None,
                 end: Optional[datetime] = None) -> List[Dict[str, Any]]:
  out: List[Dict[str, Any]] = []
  for fp in Path(LOG_DIR).glob("closed_*.json"):
    try:
      data = json.loads(fp.read_text(encoding="utf-8"))
      dt_close = _parse_dt(data.get("closed_at")) or _parse_dt(data.get("logged_at"))
      if start and dt_close and dt_close < start:
        continue
      if end and dt_close and dt_close > end:
        continue
      out.append(data)
    except Exception:
      # Bỏ qua file hỏng
      continue
  return out

# =========================
# 3) Tính KPI cơ bản trên danh sách record
# =========================

def _agg_basic(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
  n = len(recs)
  if n == 0:
    return {
      "count": 0, "win": 0, "loss": 0, "win_rate": 0.0,
      "avg_R": 0.0, "expectancy_R": 0.0, "profit_factor": 0.0,
      "max_consec_loss": 0,
      "R_sum": 0.0,
      "R_pos": 0.0,
      "R_neg": 0.0,
      "R_max": 0.0,
      "R_min": 0.0,
    }
  wins = losses = 0
  R_sum = R_pos = R_neg = 0.0
  R_max, R_min = float("-inf"), float("inf")
  max_consec_loss = cur_loss = 0
  for r in recs:
    R = float(r.get("realized_R") or 0.0)
    R_sum += R
    R_max = max(R_max, R)
    R_min = min(R_min, R)
    if R > 0:
      wins += 1
      R_pos += R
      cur_loss = 0
    elif R < 0:
      losses += 1
      R_neg += abs(R)
      cur_loss += 1
      max_consec_loss = max(max_consec_loss, cur_loss)
    else:
      # hòa: không reset chuỗi thua
      pass
  win_rate = (wins / n * 100.0) if n else 0.0
  avg_R = R_sum / n if n else 0.0
  expectancy_R = avg_R
  profit_factor = (R_pos / R_neg) if R_neg > 0 else (R_pos if R_pos > 0 else 0.0)
  return {
    "count": n,
    "win": wins,
    "loss": losses,
    "win_rate": round(win_rate, 2),
    "avg_R": round(avg_R, 3),
    "expectancy_R": round(expectancy_R, 3),
    "profit_factor": round(profit_factor, 3),
    "max_consec_loss": int(max_consec_loss),
    "R_sum": round(R_sum, 3),
    "R_pos": round(R_pos, 3),
    "R_neg": round(R_neg, 3),
    "R_max": round(R_max if R_max != float("-inf") else 0.0, 3),
    "R_min": round(R_min if R_min != float("inf") else 0.0, 3),
  }

# =========================
# 4) Tổng hợp theo kỳ + phân rã theo symbol/side/strategy
# =========================

def _window(period: str) -> Tuple[Optional[datetime], Optional[datetime]]:
  now = datetime.utcnow()
  p = (period or "all").lower()
  if p == "7d":
    return now - timedelta(days=7), now
  if p == "30d":
    return now - timedelta(days=30), now
  return None, None


def summarize(period: str = "7d") -> Dict[str, Any]:
  """
  Trả về:
  {
    period, since, until,
    overall: KPI,
    by_symbol: {SYM: KPI},
    by_side: {long|short: KPI},
    by_strategy: {strategy: KPI},
    by_symbol_side: {(SYM,SIDE): KPI},
    by_strategy_side: {(STRAT,SIDE): KPI}
  }
  """
  start, end = _window(period)
  recs = _iter_closed(start, end)

  # Overall
  overall = _agg_basic(recs)

  # Group by symbol
  by_symbol_map: Dict[str, List[Dict[str, Any]]] = {}
  for r in recs:
    sym = (r.get("symbol") or "UNKNOWN").upper()
    by_symbol_map.setdefault(sym, []).append(r)
  by_symbol = {k: _agg_basic(v) for k, v in by_symbol_map.items()}

  # Group by side
  by_side_map: Dict[str, List[Dict[str, Any]]] = {}
  for r in recs:
    sd = (r.get("side") or "").lower() or "unknown"
    by_side_map.setdefault(sd, []).append(r)
  by_side = {k: _agg_basic(v) for k, v in by_side_map.items()}

  # Group by strategy
  by_strat_map: Dict[str, List[Dict[str, Any]]] = {}
  for r in recs:
    strat = (r.get("strategy") or "unknown").lower()
    by_strat_map.setdefault(strat, []).append(r)
  by_strategy = {k: _agg_basic(v) for k, v in by_strat_map.items()}

  # Pair (symbol, side)
  ss_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
  for r in recs:
    sym = (r.get("symbol") or "UNKNOWN").upper()
    sd = (r.get("side") or "").lower() or "unknown"
    ss_map.setdefault((sym, sd), []).append(r)
  by_symbol_side = {f"{k[0]}::{k[1]}": _agg_basic(v) for k, v in ss_map.items()}

  # Pair (strategy, side)
  st_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
  for r in recs:
    strat = (r.get("strategy") or "unknown").lower()
    sd = (r.get("side") or "").lower() or "unknown"
    st_map.setdefault((strat, sd), []).append(r)
  by_strategy_side = {f"{k[0]}::{k[1]}": _agg_basic(v) for k, v in st_map.items()}

  return {
    "period": period.upper(),
    "since": (start.isoformat() + "Z") if start else None,
    "until": (end.isoformat() + "Z") if end else None,
    "overall": overall,
    "by_symbol": by_symbol,
    "by_side": by_side,
    "by_strategy": by_strategy,
    "by_symbol_side": by_symbol_side,
    "by_strategy_side": by_strategy_side,
  }

# =========================
# 5) In ra logs gọn gàng
# =========================

def summarize_print(period: str = "7d") -> None:
  s = summarize(period)
  print(f"\n==== PERFORMANCE SUMMARY [{s['period']}] ====")
  print(f"Window: {s['since']} → {s['until']}")
  o = s["overall"]
  print(
    f"Overall | n={o['count']} win={o['win']} loss={o['loss']} "
    f"win_rate={o['win_rate']}% avg_R={o['avg_R']} PF={o['profit_factor']} "
    f"max_consec_loss={o['max_consec_loss']}"
  )

  if s["by_side"]:
    print("-- By side --")
    for k, v in sorted(s["by_side"].items()):
      print(f"  {k:6s} | n={v['count']} wr={v['win_rate']}% avgR={v['avg_R']} PF={v['profit_factor']}")

  if s["by_strategy"]:
    print("-- By strategy --")
    for k, v in sorted(s["by_strategy"].items()):
      print(f"  {k:12s} | n={v['count']} wr={v['win_rate']}% avgR={v['avg_R']} PF={v['profit_factor']}")

  # Top 10 symbols by count
  if s["by_symbol"]:
    print("-- Top symbols --")
    items = sorted(s["by_symbol"].items(), key=lambda kv: (-kv[1]['count'], kv[0]))[:10]
    for k, v in items:
      print(f"  {k:10s} | n={v['count']} wr={v['win_rate']}% avgR={v['avg_R']} PF={v['profit_factor']}")

  print("==== END SUMMARY ====\n")

# =========================
# 6) CLI đơn giản
# =========================
if __name__ == "__main__":
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("--period", default="7d", choices=["7d", "30d", "all"], help="Khoảng thời gian tổng hợp")
  ap.add_argument("--print", dest="do_print", action="store_true", help="In ra console")
  args = ap.parse_args()

  if args.do_print:
    summarize_print(args.period)
  else:
    print(json.dumps(summarize(args.period), ensure_ascii=False, indent=2))
