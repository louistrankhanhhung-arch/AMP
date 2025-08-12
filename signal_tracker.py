from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from app.utils import humanize_delta
from telebot import TeleBot

@dataclass
class SignalState:
    signal: Dict[str, Any]
    chat_id: int = 0
    message_id: int = 0
    fills: List[Tuple[float, datetime]] = field(default_factory=list)  # (price, time)
    realized_R: float = 0.0
    remaining_pct: float = 100.0

class SignalTracker:
    def __init__(self, bot: TeleBot):
        self.bot = bot
        self.states: Dict[str, SignalState] = {}

    def register_post(self, signal_id: str, chat_id: int, message_id: int, signal: Dict[str, Any]):
        self.states[signal_id] = SignalState(signal=signal, chat_id=chat_id, message_id=message_id)

    def _reply(self, st: SignalState, text: str):
        self.bot.send_message(st.chat_id, text, parse_mode='HTML', reply_to_message_id=st.message_id)

    def on_entry_fill(self, signal_id: str, price: float, t: datetime):
        st = self.states[signal_id]
        st.fills.append((price, t))

    def on_tp_hit(self, signal_id: str, tp_idx: int, price: float, t: datetime):
        st = self.states[signal_id]
        sig = st.signal
        atr = sig.get('atr14', 1.0)
        sl = sig['sl']
        avg_entry = sum(p for p,_ in st.fills)/max(len(st.fills),1) if st.fills else sig['entry'][0]
        R = (price - avg_entry) / max(avg_entry - sl, 1e-9)
        part_pct = sig.get('rules',{}).get('scale_out',[{"tp_idx":0,"pct":30}])[0]['pct']
        part_frac = part_pct/100.0
        st.realized_R += R * part_frac
        st.remaining_pct = max(0.0, st.remaining_pct - part_pct)

        hold = "0m"
        if st.fills:
            hold = humanize_delta(t - st.fills[0][1])
        text = (f"<b>TP{tp_idx+1} hit</b> @ {price}\n"
                f"PnL tạm: {st.realized_R:.2f}R\n"
                f"Thời gian nắm giữ: {hold}\n"
                f"Còn lại: {st.remaining_pct:.0f}% | SL hiện tại: {sl}")
        self._reply(st, text)

    def on_stopped_or_completed(self, signal_id: str, status: str, t: datetime):
        st = self.states[signal_id]
        sig = st.signal
        hold = "0m"
        if st.fills:
            hold = humanize_delta(t - st.fills[0][1])
        text = (f"<b>Tóm tắt kèo</b> {sig['symbol']} {sig['timeframe']}\n"
                f"Kết quả: {status} | R thực: {st.realized_R:.2f}R\n"
                f"Nắm giữ: {hold}")
        self._reply(st, text)

def render_full_signal_by_id(signal_id: str) -> str:
    # TODO: fetch from DB/cache; placeholder
    return f"<b>FULL SIGNAL</b> for {signal_id} (demo)"
