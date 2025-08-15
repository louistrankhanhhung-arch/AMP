# signal_tracker.py
from __future__ import annotations
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
import math
import traceback

from notifier import Notifier, PostRef

# (tuỳ chọn) ghi log khi kèo đóng
try:
    from performance_logger import log_closed_signal  # noqa
except Exception:  # module not present
    log_closed_signal = None


def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _default_scale_out(tps_len: int) -> List[float]:
    if tps_len >= 3:
        return [0.30, 0.40, 0.30]
    if tps_len == 2:
        return [0.50, 0.50]
    return [1.00]


@dataclass
class SignalState:
    signal_id: str
    ref: PostRef
    signal: Dict[str, Any]               # raw signal: {'symbol','side','entries','stop','tps','leverage',...}
    created_at: str = field(default_factory=_utcnow)

    # runtime state
    fills: List[Dict[str, Any]] = field(default_factory=list)  # [{'idx':0,'price':..., 'ts':...}, ...]
    tps_hit: Set[int] = field(default_factory=set)
    realized_R: float = 0.0
    remaining_pct: float = 100.0
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    status: str = "OPEN"  # OPEN | CLOSED | CANCELLED
    current_sl: Optional[float] = None   # có thể dời về BE sau TP1
    sl_mode: str = "tick"                # 'tick' | 'close_4h'

    def side(self) -> str:
        return (self.signal.get("side") or "").lower()

    def symbol(self) -> str:
        return self.signal.get("symbol")

    def entries(self) -> List[float]:
        return list(self.signal.get("entries") or [])

    def stop(self) -> float:
        return float(self.signal.get("stop"))

    def tps(self) -> List[float]:
        return list(self.signal.get("tps") or [])

    def scale_out(self) -> List[float]:
        sc = self.signal.get("scale_out")
        if isinstance(sc, list) and sc and abs(sum(sc) - 1.0) < 1e-6:
            return sc
        return _default_scale_out(len(self.tps()))

    def entry_filled(self, idx: int) -> bool:
        return any(f.get("idx") == idx for f in self.fills)

    def tp_hit_flag(self, idx: int) -> bool:
        return idx in self.tps_hit

    def avg_entry(self) -> Optional[float]:
        if not self.fills:
            return None
        ps = [float(f["price"]) for f in self.fills]
        return sum(ps) / len(ps)

    def base_risk(self) -> Optional[float]:
        """denominator R theo side; >0"""
        ae = self.avg_entry()
        if ae is None:
            return None
        sl = self.current_sl if self.current_sl is not None else self.stop()
        if self.side() == "long":
            return max(ae - sl, 1e-12)
        else:
            return max(sl - ae, 1e-12)

    def is_open(self) -> bool:
        return self.status == "OPEN"

    def mark_closed(self, status: str):
        self.status = "CLOSED"
        self.closed_at = _utcnow()
        self.signal["final_status"] = status


class SignalTracker:
    def __init__(self, notifier: Notifier):
        self.notifier = notifier
        self.states: Dict[str, SignalState] = {}
        self._by_symbol: Dict[str, str] = {}  # symbol -> last signal_id

    # ---------- registration ----------
    def register_post(self, signal_id: str, ref: PostRef, signal: Dict[str, Any], sl_mode: str = "tick"):
        """
        Lưu state kèo sau khi bạn đã post lên Telegram (hay Console).
        - ref: PostRef(chat_id, message_id) nhận từ Notifier.post(...)
        - signal: {'symbol','side','entries','stop','tps','leverage', ...}
        """
        st = SignalState(signal_id=signal_id, ref=ref, signal=signal, sl_mode=sl_mode)
        st.current_sl = float(signal.get("stop"))
        self.states[signal_id] = st
        self._by_symbol[st.symbol()] = signal_id

    def get_state(self, signal_id: str) -> Optional[SignalState]:
        return self.states.get(signal_id)

    def get_state_by_symbol(self, symbol: str) -> Optional[SignalState]:
        sid = self._by_symbol.get(symbol)
        if sid:
            st = self.states.get(sid)
            if st and st.is_open():
                return st
        # fallback: tìm signal mở gần nhất của symbol
        latest: Optional[SignalState] = None
        for st in self.states.values():
            if st.symbol() == symbol and st.is_open():
                latest = st if (not latest or st.created_at > latest.created_at) else latest
        return latest

    # ---------- messaging ----------
    def _reply(self, st: SignalState, text: str):
        try:
            self.notifier.reply(st.ref, text)
        except Exception:
            print("[tracker] reply error:\n" + traceback.format_exc())

    # ---------- events ----------
    def on_entry_fill(self, signal_id: str, price: float, ts: Optional[str] = None):
        st = self.get_state(signal_id)
        if not st or not st.is_open():
            return
        # nếu entry này đã fill thì bỏ qua
        # chọn idx đầu tiên chưa fill mà điều kiện giá thỏa
        try:
            side = st.side()
            filled_any = False
            for idx, e in enumerate(st.entries()):
                if st.entry_filled(idx):
                    continue
                cond = (price <= e) if side == "long" else (price >= e)
                if cond:
                    st.fills.append({"idx": idx, "price": float(price), "ts": ts or _utcnow()})
                    if not st.opened_at:
                        st.opened_at = st.fills[-1]["ts"]
                    filled_any = True
                    self._reply(st, f"[FILLED] Entry {idx+1} @ {price}")
                    # chỉ fill một bậc mỗi tick để tránh double-logging
                    break
            if not filled_any:
                return
        except Exception:
            print("[tracker] on_entry_fill error:\n" + traceback.format_exc())

    def on_tp_hit(self, signal_id: str, tp_idx: int, price: float, ts: Optional[str] = None):
        st = self.get_state(signal_id)
        if not st or not st.is_open():
            return
        # chỉ xử lý khi đã có ít nhất 1 entry fill
        ae = st.avg_entry()
        if ae is None:
            return
        if st.tp_hit_flag(tp_idx):
            return

        # check điều kiện giá theo side
        side = st.side()
        tp_list = st.tps()
        if tp_idx < 0 or tp_idx >= len(tp_list):
            return
        tp_price = tp_list[tp_idx]
        cond = (price >= tp_price) if side == "long" else (price <= tp_price)
        if not cond:
            return

        st.tps_hit.add(tp_idx)

        # tỷ lệ thoát ở TP này
        scale = st.scale_out()
        pct_exit = scale[tp_idx] if tp_idx < len(scale) else (st.remaining_pct / 100.0)
        pct_exit = max(0.0, min(pct_exit, st.remaining_pct / 100.0))
        st.remaining_pct = max(0.0, st.remaining_pct - pct_exit * 100.0)

        # tính R
        denom = st.base_risk()
        if denom is None or denom <= 0:
            rr = 0.0
        else:
            rr = ((price - ae) / denom) if side == "long" else ((ae - price) / denom)

        st.realized_R += rr * pct_exit

        self._reply(
            st,
            f"[TP{tp_idx+1}] hit @ {price} | realized +{rr:.2f}R × {int(pct_exit*100)}% | total {st.realized_R:.2f}R | remain {st.remaining_pct:.1f}%"
        )

        # quy tắc dời SL về BE sau TP1 (tuỳ chọn)
        if tp_idx == 0:
            st.current_sl = ae  # BE

        # nếu đã thoát hết
        if st.remaining_pct <= 1e-6:
            st.mark_closed(status="TP-all")
            self._reply(st, f"[CLOSED] TP-all | total {st.realized_R:.2f}R")
            if log_closed_signal:
                try:
                    log_closed_signal({
                        "signal_id": st.signal_id, "symbol": st.symbol(), "side": side,
                        "opened_at": st.opened_at, "closed_at": st.closed_at,
                        "result": "TP-all", "realized_R": st.realized_R,
                        "entries": st.entries(), "stop": st.stop(), "tps": st.tps()
                    })
                except Exception:
                    pass

    def on_stopped_or_completed(self, signal_id: str, status: str, price: Optional[float] = None, ts: Optional[str] = None):
        """
        status: 'SL' | 'MANUAL_CLOSE' | 'CANCELLED'
        price: giá khớp khi SL/đóng tay (nếu có)
        """
        st = self.get_state(signal_id)
        if not st or not st.is_open():
            return

        ae = st.avg_entry()
        if ae is None:
            # chưa fill entry nào mà hủy kèo
            st.mark_closed(status=status)
            self._reply(st, f"[CLOSED] {status} (no fill)")
            return

        # phần còn lại đóng ở 'price' (nếu None → dùng stop hiện tại)
        px = price if (price is not None) else (st.current_sl if st.current_sl is not None else st.stop())
        denom = st.base_risk()
        if denom is None or denom <= 0:
            rr = -1.0
        else:
            if st.side() == "long":
                rr = (px - ae) / denom
            else:
                rr = (ae - px) / denom

        remaining_frac = st.remaining_pct / 100.0
        st.realized_R += rr * remaining_frac
        st.remaining_pct = 0.0

        st.mark_closed(status=status)
        self._reply(st, f"[CLOSED] {status} @ {px} | total {st.realized_R:.2f}R")

        if log_closed_signal:
            try:
                log_closed_signal({
                    "signal_id": st.signal_id, "symbol": st.symbol(), "side": st.side(),
                    "opened_at": st.opened_at, "closed_at": st.closed_at,
                    "result": status, "realized_R": st.realized_R,
                    "entries": st.entries(), "stop": st.stop(), "tps": st.tps(),
                    "closed_price": px
                })
            except Exception:
                pass
